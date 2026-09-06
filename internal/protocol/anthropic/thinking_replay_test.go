package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

// Reproducer for:
//
//	level=ERROR msg=提供商错误 model=deepseek-v4-pro status=400
//	error="The content[].thinking in the thinking mode must be passed back to the API."
//
// DeepSeek's Anthropic-compatible endpoint requires the thinking block of every
// replayed assistant turn to carry both the `thinking` and the `signature` keys.
// `omitempty` on either key drops it, and the endpoint then treats the turn as
// having no thinking block at all and answers 400.
//
// The placeholder block the deepseek_v4 extension injects for a turn whose
// thinking text was never captured must therefore serialize as
// {"type":"thinking","thinking":"","signature":""} — the shape DeepSeek accepts.
func TestThinkingBlockMarshalEmitsBothThinkingAndSignatureKeys(t *testing.T) {
	out, err := json.Marshal([]ContentBlock{{Type: "thinking"}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `[{"type":"thinking","thinking":"","signature":""}]`
	if got := string(out); got != want {
		t.Errorf("placeholder thinking block\n got: %s\nwant: %s", got, want)
	}
}

func TestThinkingBlockMarshalKeepsTextAndSignature(t *testing.T) {
	out, err := json.Marshal([]ContentBlock{{Type: "thinking", Thinking: "pondering", Signature: "sig_1"}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(out)
	for _, want := range []string{`"type":"thinking"`, `"thinking":"pondering"`, `"signature":"sig_1"`} {
		if !strings.Contains(got, want) {
			t.Errorf("marshalled thinking block missing %s: %s", want, got)
		}
	}
}

// A signature-only thinking block is replayable even though it has no visible
// thinking text; it must serialize with the signature intact.
func TestThinkingBlockMarshalSignatureOnly(t *testing.T) {
	out, err := json.Marshal([]ContentBlock{{Type: "thinking", Signature: "sig_1"}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(out)
	for _, want := range []string{`"type":"thinking"`, `"thinking":""`, `"signature":"sig_1"`} {
		if !strings.Contains(got, want) {
			t.Errorf("signature-only thinking block malformed: %s", got)
		}
	}
}

// Non-thinking blocks must keep their existing shape — in particular no
// stray "thinking" or "signature" keys on text/tool_use blocks.
func TestThinkingBlockMarshalLeavesOtherBlocksAlone(t *testing.T) {
	out, err := json.Marshal([]ContentBlock{
		{Type: "text", Text: "hello"},
		{Type: "tool_use", ID: "call_1", Name: "shell", Input: json.RawMessage(`{"cmd":"ls"}`)},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(out)
	for _, unwanted := range []string{`"thinking"`, `"signature"`, `"source"`, `"tool_use_id"`} {
		if strings.Contains(got, unwanted) {
			t.Errorf("unexpected key %s in marshalled content: %s", unwanted, got)
		}
	}
}

func TestHasThinkingPayload(t *testing.T) {
	cases := []struct {
		name  string
		block ContentBlock
		want  bool
	}{
		{"thinking with text", ContentBlock{Type: "thinking", Thinking: "hm"}, true},
		{"signature only", ContentBlock{Type: "thinking", Signature: "sig"}, true},
		{"empty placeholder", ContentBlock{Type: "thinking"}, false},
		{"text block", ContentBlock{Type: "text", Text: "hi"}, false},
	}
	for _, tc := range cases {
		if got := tc.block.HasThinkingPayload(); got != tc.want {
			t.Errorf("%s: HasThinkingPayload() = %v, want %v", tc.name, got, tc.want)
		}
	}
}
