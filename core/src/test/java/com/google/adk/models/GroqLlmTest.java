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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.sun.net.httpserver.HttpServer;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class GroqLlmTest {

  @Test
  public void getLlm_withGroqPrefix_succeeds() {
    assertThat(LlmRegistry.matchesAnyPattern("groq/llama-3.1-8b-instant")).isTrue();
    assertThat(LlmRegistry.matchesAnyPattern("groq/qwen/qwen3-32b")).isTrue();
  }

  @Test
  public void getLlm_withoutGroqPrefix_doesNotMatchGroqProvider() {
    assertThat(LlmRegistry.matchesAnyPattern("llama-3.1-8b-instant")).isFalse();
  }

  @Test
  public void isExceptionRetryable_toolUseFailed_returnsTrue() throws Exception {
    RuntimeException error =
        errorFromGroq(
            "{\"error\":{\"message\":\"Failed to parse tool call arguments as JSON\","
                + "\"type\":\"invalid_request_error\",\"code\":\"tool_use_failed\","
                + "\"failed_generation\":\"{bad\"}}");

    assertThat(groqLlm().isExceptionRetryable(error, Set.of())).isTrue();
  }

  @Test
  public void isExceptionRetryable_otherBadRequest_returnsFalse() throws Exception {
    RuntimeException error =
        errorFromGroq(
            "{\"error\":{\"message\":\"bad request\","
                + "\"type\":\"invalid_request_error\",\"code\":\"invalid_value\"}}");

    assertThat(groqLlm().isExceptionRetryable(error, Set.of())).isFalse();
  }

  private static GroqLlm groqLlm() {
    return GroqLlm.builder()
        .modelName("groq/openai/gpt-oss-120b")
        .apiClient(OpenAIOkHttpClient.builder().apiKey("test").build())
        .build();
  }

  /** Sends one request to a local server that answers 400 with {@code body}, as Groq does. */
  private static RuntimeException errorFromGroq(String body) throws Exception {
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/",
        exchange -> {
          byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
          exchange.getResponseHeaders().add("Content-Type", "application/json");
          exchange.sendResponseHeaders(400, bytes.length);
          try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
          }
        });
    server.start();
    try {
      GroqLlm llm =
          GroqLlm.builder()
              .modelName("groq/openai/gpt-oss-120b")
              .apiClient(
                  OpenAIOkHttpClient.builder()
                      .apiKey("test")
                      .baseUrl("http://127.0.0.1:" + server.getAddress().getPort())
                      .maxRetries(0)
                      .build())
              .build();
      LlmRequest request =
          LlmRequest.builder()
              .contents(
                  ImmutableList.of(
                      Content.builder().role("user").parts(Part.fromText("hi")).build()))
              .build();
      return assertThrows(
          RuntimeException.class, () -> llm.generateContent(request, false).blockingFirst());
    } finally {
      server.stop(0);
    }
  }
}
