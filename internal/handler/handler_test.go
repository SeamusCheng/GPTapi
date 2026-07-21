package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"aurora/internal/accounts"
	"aurora/internal/config"
	officialtypes "aurora/typings/official"

	"github.com/gin-gonic/gin"
)

// ─── Test: writeChatCompletionStreamDone ─────────────────────────

func TestWriteChatCompletionStreamDoneAddsStopBeforeDone(t *testing.T) {
	gin.SetMode(gin.TestMode)
	writer := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(writer)

	writeChatCompletionStreamDone(c, false, "auto", "conv-xxx")

	lines := sseDataLines(writer.Body.String())
	if len(lines) != 2 {
		t.Fatalf("data line count = %d, want 2; output: %s", len(lines), writer.Body.String())
	}
	var stopChunk map[string]interface{}
	if err := json.Unmarshal([]byte(lines[0]), &stopChunk); err != nil {
		t.Fatalf("invalid stop chunk: %v", err)
	}
	if stopChunk["conversation_id"] != "conv-xxx" {
		t.Fatalf("conversation_id = %#v, want conv-xxx", stopChunk["conversation_id"])
	}
	choices := stopChunk["choices"].([]interface{})
	if choices[0].(map[string]interface{})["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %#v, want stop", choices[0].(map[string]interface{})["finish_reason"])
	}
	if lines[1] != "[DONE]" {
		t.Fatalf("last data line = %q, want [DONE]", lines[1])
	}
}

func TestWriteChatCompletionStreamDoneSkipsDuplicateStop(t *testing.T) {
	gin.SetMode(gin.TestMode)
	writer := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(writer)

	writeChatCompletionStreamDone(c, true, "auto", "conv-xxx")

	lines := sseDataLines(writer.Body.String())
	if len(lines) != 1 || lines[0] != "[DONE]" {
		t.Fatalf("data lines = %#v, want only [DONE]", lines)
	}
}

// ─── Test: toolCallingEnabled ────────────────────────────────────

func TestWriteToolCallingStreamResponseWithText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	writer := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(writer)

	writeToolCallingStreamResponse(c, "HERMES_OK", nil, 10, 2, "auto", "conv-text", &officialtypes.StreamOptions{IncludeUsage: true})

	lines := sseDataLines(writer.Body.String())
	if len(lines) != 5 {
		t.Fatalf("data line count = %d, want 5; output: %s", len(lines), writer.Body.String())
	}
	if !strings.Contains(lines[1], `"content":"HERMES_OK"`) {
		t.Fatalf("content chunk missing: %s", lines[1])
	}
	if !strings.Contains(lines[2], `"finish_reason":"stop"`) {
		t.Fatalf("stop chunk missing: %s", lines[2])
	}
	if !strings.Contains(lines[3], `"total_tokens":12`) || lines[4] != "[DONE]" {
		t.Fatalf("usage or done chunk missing: %#v", lines)
	}
}

func TestWriteToolCallingStreamResponseWithToolCall(t *testing.T) {
	gin.SetMode(gin.TestMode)
	writer := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(writer)
	calls := []officialtypes.ToolCall{{
		ID:   "call-1",
		Type: "function",
		Function: officialtypes.ToolCallFunc{
			Name:      "read_file",
			Arguments: `{"path":"README.md"}`,
		},
	}}

	writeToolCallingStreamResponse(c, "", calls, 10, 3, "auto", "conv-tool", nil)

	lines := sseDataLines(writer.Body.String())
	if len(lines) != 4 {
		t.Fatalf("data line count = %d, want 4; output: %s", len(lines), writer.Body.String())
	}
	if !strings.Contains(lines[1], `"tool_calls"`) || !strings.Contains(lines[1], `"name":"read_file"`) {
		t.Fatalf("tool call chunk missing: %s", lines[1])
	}
	if !strings.Contains(lines[2], `"finish_reason":"tool_calls"`) || lines[3] != "[DONE]" {
		t.Fatalf("tool stop or done chunk missing: %#v", lines)
	}
}

func TestToolCallingEnabledFromConfig(t *testing.T) {
	okCfg := &config.Config{ToolCallingEnabled: true}
	disabledCfg := &config.Config{ToolCallingEnabled: false}

	if toolCallingEnabled(nil, okCfg) {
		t.Error("toolCallingEnabled(nil, true) should be false (len(nil)==0)")
	}
	if toolCallingEnabled(nil, disabledCfg) {
		t.Error("toolCallingEnabled(nil, false) should be false")
	}
	// empty tools slice with config enabled → false
	if toolCallingEnabled([]officialtypes.Tool{}, okCfg) {
		t.Error("toolCallingEnabled([], true) should be false")
	}
	// with actual tools and config enabled → true
	tools := []officialtypes.Tool{{Type: "function", Function: officialtypes.ToolFunction{Name: "test"}}}
	if !toolCallingEnabled(tools, okCfg) {
		t.Error("toolCallingEnabled([tool], true) should be true")
	}
}

// ─── Test: original_requestHasFiles ──────────────────────────────

func TestOriginalRequestHasFiles(t *testing.T) {
	req := officialtypes.APIRequest{
		Messages: []officialtypes.APIMessage{
			{
				Role:    "user",
				Content: officialtypes.MessageContent{TextValue: "hello"},
			},
		},
	}
	if original_requestHasFiles(req) {
		t.Error("should be false when no files")
	}
}

// ─── Test: countMessagesTokens ───────────────────────────────────

func TestCountMessagesTokens(t *testing.T) {
	zero := countMessagesTokens(nil)
	if zero != 0 {
		t.Errorf("nil messages should return 0, got %d", zero)
	}
}

// ─── Test: resolveAccount ────────────────────────────────────────

func TestResolveAccountEmptyPool(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	pool := accounts.NewPool(nil)
	cfg := &config.Config{}

	acct, _, err := resolveAccount(c, pool, cfg, false)
	if err == nil {
		t.Fatal("expected error with empty pool")
	}
	if acct != nil {
		t.Fatal("expected nil account")
	}
}

func TestResolveAccountWithGlobalKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Set("Authorization", "Bearer my-global-key")

	pool := accounts.NewPool(nil)
	acct := accounts.NewAccount("test", accounts.TypeFree, "test-token")
	acct.Status = accounts.StatusActive
	pool.AddAccount(acct)
	cfg := &config.Config{Authorization: "my-global-key"}

	result, _, err := resolveAccount(c, pool, cfg, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected account, got nil")
	}
	if result.Token != "test-token" {
		t.Errorf("got token %q, want test-token", result.Token)
	}
}

// ─── helpers ─────────────────────────────────────────────────────

func sseDataLines(output string) []string {
	var lines []string
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		lines = append(lines, strings.TrimPrefix(line, "data: "))
	}
	return lines
}
