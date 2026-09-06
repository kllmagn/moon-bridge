package server

import (
	"encoding/json"
	"strings"
	"testing"

	deepseekv4 "moonbridge/internal/extension/deepseek_v4"
	"moonbridge/internal/extension/plugin"
	"moonbridge/internal/format"
	"moonbridge/internal/protocol/anthropic"
	openai "moonbridge/internal/protocol/openai"
	"moonbridge/internal/service/provider"
	"moonbridge/internal/service/stats"
	"moonbridge/internal/session"
)

func TestRequestHasImage(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"empty input", "", false},
		{"null input", "null", false},
		{"string input", `"hello"`, false},
		{"array without image", `[{"type":"input_text","text":"hello"}]`, false},
		{"array with input_image", `[{"type":"input_text","text":"hello"},{"type":"input_image","image_url":"data:image/png;base64,abc"}]`, true},
		{"array with image", `[{"type":"text","text":"hello"},{"type":"image","image_url":"data:image/png;base64,abc"}]`, true},
		{"array with image_url", `[{"type":"image_url","image_url":"https://example.com/img.png"}]`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := requestHasImage(json.RawMessage(tt.input))
			if got != tt.want {
				t.Errorf("requestHasImage(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestHasModalityImage(t *testing.T) {
	tests := []struct {
		name       string
		modalities []string
		want       bool
	}{
		{"nil list", nil, false},
		{"empty list", []string{}, false},
		{"only text", []string{"text"}, false},
		{"with image", []string{"text", "image"}, true},
		{"only image", []string{"image"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasModalityImage(tt.modalities)
			if got != tt.want {
				t.Errorf("hasModalityImage(%v) = %v, want %v", tt.modalities, got, tt.want)
			}
		})
	}
}

func TestFilterCandidatesByInputNoProviderMgr(t *testing.T) {
	srv := &Server{providerMgr: nil}
	candidates := []provider.ProviderCandidate{
		{ProviderKey: "p1", UpstreamModel: "model-a"},
	}
	filtered, _ := srv.filterCandidatesByInput(candidates, json.RawMessage(`[{"type":"input_image","image_url":"data:image/png;base64,abc"}]`))
	if len(filtered) != 1 {
		t.Fatalf("without providerMgr, should return unchanged: got %d", len(filtered))
	}
}

func TestComputeCostWithProviderPricingNilStats(t *testing.T) {
	cost := computeCostWithProviderPricing(nil, nil, "model", "model", "provider", stats.BillingUsage{})
	if cost != 0 {
		t.Fatalf("nil stats should return 0, got %f", cost)
	}
}

func TestPrependCachedThinkingRejectsMissingThinkingWhenToolsPresent(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	sess.InitExtensions(map[string]any{
		"deepseek_v4": state,
	})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "text", Text: "plain assistant text"},
				},
			},
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "tool_use", ID: "call-miss", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	err := prependCachedThinking(req, sess)
	if err == nil {
		t.Fatal("missing thinking should be rejected before the provider call")
	}
	if !strings.Contains(err.Error(), "assistant message 0") {
		t.Fatalf("unexpected replay error: %v", err)
	}
}

func TestPrependCachedThinkingRestoresAssistantTextWhenToolsPresent(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	state.RememberForAssistantText("plain assistant text", format.CoreContentBlock{
		Type:               "reasoning",
		ReasoningText:      "cached text reasoning",
		ReasoningSignature: "sig-text",
	})
	sess.InitExtensions(map[string]any{"deepseek_v4": state})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{{
			Role:    "assistant",
			Content: []anthropic.ContentBlock{{Type: "text", Text: "plain assistant text"}},
		}},
	}

	prependCachedThinking(req, sess)

	content := req.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("assistant text should receive cached thinking, got %+v", content)
	}
	if content[0].Type != "thinking" || content[0].Thinking != "cached text reasoning" || content[0].Signature != "sig-text" {
		t.Fatalf("cached thinking mismatch: %+v", content[0])
	}
}

func TestPrependCachedThinkingSkipsReplayWithoutTools(t *testing.T) {
	sess := session.New()
	sess.InitExtensions(map[string]any{"deepseek_v4": deepseekv4.NewState()})
	req := &anthropic.MessageRequest{
		Messages: []anthropic.Message{{
			Role:    "assistant",
			Content: []anthropic.ContentBlock{{Type: "text", Text: "plain assistant text"}},
		}},
	}

	if err := prependCachedThinking(req, sess); err != nil {
		t.Fatalf("no-tools request should not require replay: %v", err)
	}
	if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Type != "text" {
		t.Fatalf("no-tools request was modified: %+v", req.Messages[0].Content)
	}
}

func TestDeepseekReplayEnabledIsModelScoped(t *testing.T) {
	registry := plugin.NewRegistry(nil)
	registry.Register(deepseekv4.NewPlugin(func(model string) bool {
		return model == "deepseek-v4-flash"
	}))

	if deepseekReplayEnabled(registry, "claude-sonnet") {
		t.Fatal("DeepSeek replay should be disabled for unrelated models")
	}
	if !deepseekReplayEnabled(registry, "deepseek-v4-flash") {
		t.Fatal("DeepSeek replay should be enabled for the configured model")
	}
}

func TestPrependCachedThinkingChecksAllToolUseBlocks(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	state.RememberForToolCalls(
		[]string{"call-hit"},
		format.CoreContentBlock{
			Type:               "reasoning",
			ReasoningText:      "replayed",
			ReasoningSignature: "sig-hit",
		},
	)
	sess.InitExtensions(map[string]any{
		"deepseek_v4": state,
	})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "tool_use", ID: "call-miss", Name: "exec_command", Input: json.RawMessage(`{}`)},
					{Type: "tool_use", ID: "call-hit", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) < 3 {
		t.Fatalf("assistant message should prepend cached thinking, got %+v", req.Messages[0].Content)
	}
	head := req.Messages[0].Content[0]
	if head.Type != "thinking" || head.Thinking != "replayed" || head.Signature != "sig-hit" {
		t.Fatalf("cached thinking block mismatch, got %+v", head)
	}
}

func TestRememberAdapterResponseContentCachesDeepSeekThinkingForLaterReplay(t *testing.T) {
	registry := plugin.NewRegistry(nil)
	dsPlugin := deepseekv4.NewPlugin(func(string) bool { return true })
	registry.Register(dsPlugin)
	if err := registry.InitAll(nil); err != nil {
		t.Fatalf("InitAll() error = %v", err)
	}

	sess := session.New()
	sess.InitExtensions(registry.NewSessionData())

	resp := &format.CoreResponse{
		Messages: []format.CoreMessage{{
			Role: "assistant",
			Content: []format.CoreContentBlock{
				{
					Type:               "reasoning",
					ReasoningText:      "trace reasoning",
					ReasoningSignature: "sig-trace",
				},
				{
					Type:      "tool_use",
					ToolUseID: "call-trace",
					ToolName:  "exec_command",
					ToolInput: json.RawMessage(`{"cmd":"pwd"}`),
				},
				{
					Type: "text",
					Text: "assistant tool turn",
				},
			},
		}},
	}

	rememberAdapterResponseContent(registry, sess, "deepseek-v4-flash", resp)

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{{
			Role: "assistant",
			Content: []anthropic.ContentBlock{
				{Type: "tool_use", ID: "call-trace", Name: "exec_command", Input: json.RawMessage(`{"cmd":"pwd"}`)},
			},
		}},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) < 2 {
		t.Fatalf("assistant message should prepend cached thinking, got %+v", req.Messages[0].Content)
	}
	head := req.Messages[0].Content[0]
	if head.Type != "thinking" || head.Thinking != "trace reasoning" || head.Signature != "sig-trace" {
		t.Fatalf("prepended thinking block mismatch, got %+v", head)
	}
}

func TestStreamReplayCachesDeepSeekThinkingForLaterReplay(t *testing.T) {
	registry := plugin.NewRegistry(nil)
	dsPlugin := deepseekv4.NewPlugin(func(string) bool { return true })
	registry.Register(dsPlugin)
	if err := registry.InitAll(nil); err != nil {
		t.Fatalf("InitAll() error = %v", err)
	}

	sess := session.New()
	sess.InitExtensions(registry.NewSessionData())

	states := registry.NewStreamStates("deepseek-v4-flash")
	if states == nil {
		t.Fatal("NewStreamStates() returned nil")
	}

	streamEvents := []plugin.StreamEvent{
		{
			Type:  "block_start",
			Index: 0,
			Block: &format.CoreContentBlock{
				Type:               "reasoning",
				ReasoningText:      "",
				ReasoningSignature: "",
			},
		},
		{
			Type:  "block_delta",
			Index: 0,
			Delta: anthropic.StreamDelta{
				Type:     "thinking_delta",
				Thinking: "stream reasoning",
			},
		},
		{
			Type:  "block_delta",
			Index: 0,
			Delta: anthropic.StreamDelta{
				Type:      "signature_delta",
				Signature: "sig-stream",
			},
		},
		{
			Type:  "block_stop",
			Index: 0,
		},
		{
			Type:  "block_start",
			Index: 1,
			Block: &format.CoreContentBlock{
				Type:      "tool_use",
				ToolUseID: "call-stream",
				ToolName:  "exec_command",
			},
		},
	}

	for _, ev := range streamEvents {
		registry.OnStreamEvent("deepseek-v4-flash", ev, states)
	}
	registry.OnStreamComplete("deepseek-v4-flash", states, "", sess.ExtensionData)

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{{
			Role: "assistant",
			Content: []anthropic.ContentBlock{
				{Type: "tool_use", ID: "call-stream", Name: "exec_command", Input: json.RawMessage(`{"cmd":"pwd"}`)},
			},
		}},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) < 2 {
		t.Fatalf("assistant message should prepend cached thinking, got %+v", req.Messages[0].Content)
	}
	head := req.Messages[0].Content[0]
	if head.Type != "thinking" || head.Thinking != "stream reasoning" || head.Signature != "sig-stream" {
		t.Fatalf("prepended stream thinking block mismatch, got %+v", head)
	}
}

func TestRememberStreamResponseContentCachesDeepSeekThinkingForLaterReplay(t *testing.T) {
	registry := plugin.NewRegistry(nil)
	dsPlugin := deepseekv4.NewPlugin(func(string) bool { return true })
	registry.Register(dsPlugin)
	if err := registry.InitAll(nil); err != nil {
		t.Fatalf("InitAll() error = %v", err)
	}

	sess := session.New()
	sess.InitExtensions(registry.NewSessionData())

	streamResp := &openai.Response{
		Output: []openai.OutputItem{
			{
				Type:   "reasoning",
				Status: "completed",
				Summary: []openai.ReasoningItemSummary{{
					Type:      "text",
					Text:      "trace stream reasoning",
					Signature: "sig-trace-stream",
				}},
			},
			{
				Type:      "function_call",
				CallID:    "call-stream-trace",
				Name:      "exec_command",
				Arguments: `{"cmd":"pwd"}`,
				Status:    "completed",
			},
		},
	}

	if !rememberStreamResponseContent(registry, sess, "deepseek-v4-flash", streamResp) {
		t.Fatal("rememberStreamResponseContent() = false, want true")
	}

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{{
			Role: "assistant",
			Content: []anthropic.ContentBlock{
				{Type: "tool_use", ID: "call-stream-trace", Name: "exec_command", Input: json.RawMessage(`{"cmd":"pwd"}`)},
			},
		}},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) < 2 {
		t.Fatalf("assistant message should prepend cached thinking, got %+v", req.Messages[0].Content)
	}
	head := req.Messages[0].Content[0]
	if head.Type != "thinking" || head.Thinking != "trace stream reasoning" || head.Signature != "sig-trace-stream" {
		t.Fatalf("prepended stream-response thinking block mismatch, got %+v", head)
	}
}

// The placeholder thinking block injected on an earlier turn is not real
// reasoning. If it counts as satisfying the replay check, a later turn never
// replays the genuine thinking block the provider asked for, and the request
// fails with 400 "content[].thinking ... must be passed back to the API".
// A cache hit must replace the placeholder, not stack behind it.
func TestPrependCachedThinkingReplacesPlaceholderWithCachedThinking(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	state.RememberForToolCalls(
		[]string{"call-1"},
		format.CoreContentBlock{
			Type:               "reasoning",
			ReasoningText:      "real thinking",
			ReasoningSignature: "sig-1",
		},
	)
	sess.InitExtensions(map[string]any{"deepseek_v4": state})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					// Degenerate placeholder carried in from a previous request.
					{Type: "thinking"},
					{Type: "tool_use", ID: "call-1", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	content := req.Messages[0].Content
	thinking := 0
	for _, block := range content {
		if block.Type != "thinking" {
			continue
		}
		thinking++
		if block.Thinking != "real thinking" || block.Signature != "sig-1" {
			t.Fatalf("placeholder not replaced with cached thinking, got %+v", content)
		}
	}
	if thinking != 1 {
		t.Fatalf("got %d thinking blocks, want exactly 1: %+v", thinking, content)
	}
}

// A turn that already carries replayable thinking is left exactly as-is.
func TestPrependCachedThinkingLeavesReplayableThinkingAlone(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	state.RememberForToolCalls([]string{"call-1"}, format.CoreContentBlock{
		Type:               "reasoning",
		ReasoningText:      "should not be used",
		ReasoningSignature: "other-sig",
	})
	sess.InitExtensions(map[string]any{"deepseek_v4": state})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "thinking", Thinking: "original", Signature: "sig-original"},
					{Type: "tool_use", ID: "call-1", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	content := req.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("content should be untouched, got %+v", content)
	}
	if content[0].Thinking != "original" || content[0].Signature != "sig-original" {
		t.Fatalf("existing thinking block overwritten: %+v", content[0])
	}
}

// A signature-only thinking block counts as replayable and must survive.
func TestPrependCachedThinkingKeepsSignatureOnlyBlock(t *testing.T) {
	sess := session.New()
	sess.InitExtensions(map[string]any{"deepseek_v4": deepseekv4.NewState()})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "thinking", Signature: "sig-only"},
					{Type: "tool_use", ID: "call-1", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	content := req.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("signature-only thinking should satisfy replay, got %+v", content)
	}
	if content[0].Type != "thinking" || content[0].Signature != "sig-only" {
		t.Fatalf("signature-only block damaged: %+v", content[0])
	}
}

// Replaying thinking into a request that switched thinking mode off is a
// protocol mismatch, so an explicit "disabled" turns replay off.
func TestPrependCachedThinkingSkippedWhenThinkingDisabled(t *testing.T) {
	sess := session.New()
	sess.InitExtensions(map[string]any{"deepseek_v4": deepseekv4.NewState()})

	req := &anthropic.MessageRequest{
		Thinking: &anthropic.ThinkingConfig{Type: "disabled"},
		Tools:    []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "tool_use", ID: "call-1", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) != 1 {
		t.Fatalf("no thinking block should be injected when thinking is disabled, got %+v", req.Messages[0].Content)
	}
}

func TestPrependCachedThinkingDoesNotFabricatePlaceholder(t *testing.T) {
	sess := session.New()
	sess.InitExtensions(map[string]any{"deepseek_v4": deepseekv4.NewState()})

	req := &anthropic.MessageRequest{
		Tools: []anthropic.Tool{{Name: "exec_command", InputSchema: map[string]any{"type": "object"}}},
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "tool_use", ID: "call-1", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	err := prependCachedThinking(req, sess)
	if err == nil {
		t.Fatal("missing thinking should be rejected instead of replaced with a placeholder")
	}
	if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Type != "tool_use" {
		t.Fatalf("request was modified after replay failure: %+v", req.Messages[0].Content)
	}
}
