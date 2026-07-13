/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.adk.models;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import com.google.genai.types.Tool;
import com.google.genai.types.Type;
import io.github.ollama4j.Ollama;
import io.github.ollama4j.models.chat.OllamaChatMessage;
import io.github.ollama4j.models.chat.OllamaChatMessageRole;
import io.github.ollama4j.models.chat.OllamaChatRequest;
import io.github.ollama4j.models.chat.OllamaChatResponseModel;
import io.github.ollama4j.models.chat.OllamaChatResult;
import io.github.ollama4j.models.chat.OllamaChatTokenHandler;
import io.github.ollama4j.models.chat.OllamaChatToolCalls;
import io.github.ollama4j.tools.OllamaToolCallsFunction;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class OllamaLlmTest {

  @Test
  public void registry_matchesOllamaPrefixedModels() {
    assertThat(LlmRegistry.matchesAnyPattern("ollama/llama3.1")).isTrue();
    assertThat(LlmRegistry.matchesAnyPattern("llama3.1")).isFalse();
  }

  @Test
  public void generateContent_stripsRegistryPrefixAndConvertsResponse() throws Exception {
    CapturingOllama api = new CapturingOllama("hello from ollama");
    OllamaLlm llm = OllamaLlm.builder().modelName("ollama/llama3.1").apiClient(api).build();

    LlmResponse response = llm.generateContent(textRequest("hi"), false).blockingFirst();

    assertThat(api.capturedRequest.getModel()).isEqualTo("llama3.1");
    assertThat(response.content().get().role().get()).isEqualTo("model");
    assertThat(response.content().get().parts().get().get(0).text().get())
        .isEqualTo("hello from ollama");
  }

  @Test
  public void generateContent_includesSystemUserAndAssistantMessages() throws Exception {
    CapturingOllama api = new CapturingOllama("ok");
    OllamaLlm llm = OllamaLlm.builder().modelName("ollama/mistral").apiClient(api).build();

    LlmRequest request =
        LlmRequest.builder()
            .model("ollama/mistral")
            .config(
                GenerateContentConfig.builder()
                    .systemInstruction(Content.builder().parts(Part.fromText("be concise")).build())
                    .build())
            .contents(
                ImmutableList.of(
                    Content.builder().role("user").parts(Part.fromText("question")).build(),
                    Content.builder()
                        .role("model")
                        .parts(Part.fromText("previous answer"))
                        .build()))
            .build();

    llm.generateContent(request, false).blockingFirst();

    assertThat(api.capturedRequest.getModel()).isEqualTo("mistral");
    List<OllamaChatMessage> messages = api.capturedRequest.getMessages();
    assertThat(messages).hasSize(3);
    assertThat(messages.get(0).getRole()).isEqualTo(OllamaChatMessageRole.SYSTEM);
    assertThat(messages.get(0).getResponse()).isEqualTo("be concise");
    assertThat(messages.get(1).getRole()).isEqualTo(OllamaChatMessageRole.USER);
    assertThat(messages.get(1).getResponse()).isEqualTo("question");
    assertThat(messages.get(2).getRole()).isEqualTo(OllamaChatMessageRole.ASSISTANT);
    assertThat(messages.get(2).getResponse()).isEqualTo("previous answer");
  }

  @Test
  public void readTimeoutSeconds_usesSystemProperty() throws Exception {
    String previous = System.getProperty("ollama.request-timeout-seconds");
    try {
      System.setProperty("ollama.request-timeout-seconds", "1234");
      java.lang.reflect.Method method = OllamaLlm.class.getDeclaredMethod("readTimeoutSeconds");
      method.setAccessible(true);

      assertThat((Long) method.invoke(null)).isEqualTo(1234L);
    } finally {
      if (previous == null) {
        System.clearProperty("ollama.request-timeout-seconds");
      } else {
        System.setProperty("ollama.request-timeout-seconds", previous);
      }
    }
  }

  @Test
  public void generateContent_convertsToolsAndToolCalls() throws Exception {
    CapturingOllama api =
        new CapturingOllama(
            new OllamaChatMessage(
                OllamaChatMessageRole.ASSISTANT,
                "",
                null,
                ImmutableList.of(
                    new OllamaChatToolCalls(
                        "call_1",
                        new OllamaToolCallsFunction("lookupOrder", Map.of("orderId", "ORD-1")))),
                null));
    OllamaLlm llm = OllamaLlm.builder().modelName("ollama/qwen3:8b").apiClient(api).build();

    LlmRequest request =
        LlmRequest.builder()
            .model("ollama/qwen3:8b")
            .config(
                GenerateContentConfig.builder()
                    .tools(
                        ImmutableList.of(
                            Tool.builder()
                                .functionDeclarations(
                                    ImmutableList.of(
                                        FunctionDeclaration.builder()
                                            .name("lookupOrder")
                                            .description("Looks up an order.")
                                            .parameters(
                                                Schema.builder()
                                                    .type(Type.Known.OBJECT)
                                                    .properties(
                                                        Map.of(
                                                            "orderId",
                                                            Schema.builder()
                                                                .type(Type.Known.STRING)
                                                                .description("Order id.")
                                                                .build()))
                                                    .required("orderId")
                                                    .build())
                                            .build()))
                                .build()))
                    .build())
            .contents(
                ImmutableList.of(
                    Content.builder()
                        .role("user")
                        .parts(Part.fromText("Find order ORD-1"))
                        .build()))
            .build();

    LlmResponse response = llm.generateContent(request, false).blockingFirst();

    assertThat(api.capturedRequest.isUseTools()).isFalse();
    assertThat(api.capturedRequest.getTools()).hasSize(1);
    assertThat(api.capturedRequest.getTools().get(0).getToolSpec().getName())
        .isEqualTo("lookupOrder");
    assertThat(api.capturedRequest.getTools().get(0).getToolSpec().getParameters().getRequired())
        .containsExactly("orderId");
    assertThat(response.content().get().parts().get()).hasSize(1);
    FunctionCall functionCall = response.content().get().parts().get().get(0).functionCall().get();
    assertThat(functionCall.id()).hasValue("call_1");
    assertThat(functionCall.name()).hasValue("lookupOrder");
    assertThat(functionCall.args().get()).containsEntry("orderId", "ORD-1");
  }

  @Test
  public void generateContent_convertsFunctionCallAndResponseHistory() throws Exception {
    CapturingOllama api = new CapturingOllama("done");
    OllamaLlm llm = OllamaLlm.builder().modelName("ollama/qwen3:8b").apiClient(api).build();

    LlmRequest request =
        LlmRequest.builder()
            .model("ollama/qwen3:8b")
            .contents(
                ImmutableList.of(
                    Content.builder()
                        .role("model")
                        .parts(
                            Part.builder()
                                .functionCall(
                                    FunctionCall.builder()
                                        .id("call_1")
                                        .name("lookupOrder")
                                        .args(Map.of("orderId", "ORD-1"))
                                        .build())
                                .build())
                        .build(),
                    Content.builder()
                        .role("user")
                        .parts(
                            Part.builder()
                                .functionResponse(
                                    FunctionResponse.builder()
                                        .id("call_1")
                                        .name("lookupOrder")
                                        .response(Map.of("status", "shipped"))
                                        .build())
                                .build())
                        .build()))
            .build();

    llm.generateContent(request, false).blockingFirst();

    assertThat(api.capturedRequest.getMessages()).hasSize(2);
    OllamaChatMessage callMessage = api.capturedRequest.getMessages().get(0);
    assertThat(callMessage.getRole()).isEqualTo(OllamaChatMessageRole.ASSISTANT);
    assertThat(callMessage.getToolCalls()).hasSize(1);
    assertThat(callMessage.getToolCalls().get(0).getId()).isEqualTo("call_1");
    assertThat(callMessage.getToolCalls().get(0).getFunction().getName()).isEqualTo("lookupOrder");
    OllamaChatMessage responseMessage = api.capturedRequest.getMessages().get(1);
    assertThat(responseMessage.getRole()).isEqualTo(OllamaChatMessageRole.TOOL);
    assertThat(responseMessage.getResponse()).contains("\"status\":\"shipped\"");
  }

  private static OllamaChatResult chatResult(String responseText) {
    OllamaChatResponseModel responseModel = new OllamaChatResponseModel();
    responseModel.setMessage(new OllamaChatMessage(OllamaChatMessageRole.ASSISTANT, responseText));
    return new OllamaChatResult(responseModel, new java.util.ArrayList<>());
  }

  private static OllamaChatResult chatResult(OllamaChatMessage message) {
    OllamaChatResponseModel responseModel = new OllamaChatResponseModel();
    responseModel.setMessage(message);
    return new OllamaChatResult(responseModel, new java.util.ArrayList<>());
  }

  private static LlmRequest textRequest(String text) {
    return LlmRequest.builder()
        .model("ollama/llama3.1")
        .contents(
            ImmutableList.of(Content.builder().role("user").parts(Part.fromText(text)).build()))
        .build();
  }

  private static final class CapturingOllama extends Ollama {
    private final String responseText;
    private final OllamaChatMessage responseMessage;
    private OllamaChatRequest capturedRequest;

    private CapturingOllama(String responseText) {
      super("http://localhost:11434");
      this.responseText = responseText;
      this.responseMessage = null;
    }

    private CapturingOllama(OllamaChatMessage responseMessage) {
      super("http://localhost:11434");
      this.responseText = null;
      this.responseMessage = responseMessage;
    }

    @Override
    public OllamaChatResult chat(OllamaChatRequest request, OllamaChatTokenHandler tokenHandler) {
      this.capturedRequest = request;
      return responseMessage == null ? chatResult(responseText) : chatResult(responseMessage);
    }
  }
}
