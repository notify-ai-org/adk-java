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
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import io.github.amithkoujalgi.ollama4j.core.OllamaAPI;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessage;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessageRole;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatResult;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

@RunWith(JUnit4.class)
public final class OllamaLlmTest {

  @Test
  public void registry_matchesOllamaPrefixedModels() {
    assertThat(LlmRegistry.matchesAnyPattern("ollama/llama3.1")).isTrue();
    assertThat(LlmRegistry.matchesAnyPattern("llama3.1")).isFalse();
  }

  @Test
  public void generateContent_stripsRegistryPrefixAndConvertsResponse() throws Exception {
    OllamaAPI api = Mockito.mock(OllamaAPI.class);
    when(api.chat(eq("llama3.1"), anyList()))
        .thenReturn(new OllamaChatResult("hello from ollama", 12, 200, new ArrayList<>()));
    OllamaLlm llm = OllamaLlm.builder().modelName("ollama/llama3.1").apiClient(api).build();

    LlmResponse response = llm.generateContent(textRequest("hi"), false).blockingFirst();

    verify(api).chat(eq("llama3.1"), anyList());
    assertThat(response.content().get().role().get()).isEqualTo("model");
    assertThat(response.content().get().parts().get().get(0).text().get())
        .isEqualTo("hello from ollama");
  }

  @Test
  public void generateContent_includesSystemUserAndAssistantMessages() throws Exception {
    OllamaAPI api = Mockito.mock(OllamaAPI.class);
    when(api.chat(eq("mistral"), anyList()))
        .thenReturn(new OllamaChatResult("ok", 12, 200, new ArrayList<>()));
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

    ArgumentCaptor<List<OllamaChatMessage>> messagesCaptor = ArgumentCaptor.forClass(List.class);
    verify(api).chat(eq("mistral"), messagesCaptor.capture());
    List<OllamaChatMessage> messages = messagesCaptor.getValue();
    assertThat(messages).hasSize(3);
    assertThat(messages.get(0).getRole()).isEqualTo(OllamaChatMessageRole.SYSTEM);
    assertThat(messages.get(0).getContent()).isEqualTo("be concise");
    assertThat(messages.get(1).getRole()).isEqualTo(OllamaChatMessageRole.USER);
    assertThat(messages.get(1).getContent()).isEqualTo("question");
    assertThat(messages.get(2).getRole()).isEqualTo(OllamaChatMessageRole.ASSISTANT);
    assertThat(messages.get(2).getContent()).isEqualTo("previous answer");
  }

  private static LlmRequest textRequest(String text) {
    return LlmRequest.builder()
        .model("ollama/llama3.1")
        .contents(
            ImmutableList.of(Content.builder().role("user").parts(Part.fromText(text)).build()))
        .build();
  }
}
