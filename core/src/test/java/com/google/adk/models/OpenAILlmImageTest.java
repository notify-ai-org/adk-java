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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.sun.net.httpserver.HttpServer;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class OpenAILlmImageTest {

  private static final byte[] PNG = {(byte) 0x89, 'P', 'N', 'G', 1, 2, 3};

  @Test
  public void userImagesAreSentAsImageContentPartsInOrder() throws Exception {
    JsonNode content =
        sentUserContent(
            Content.builder()
                .role("user")
                .parts(
                    Part.fromText("Extract brand values."),
                    Part.fromBytes(PNG, "image/png"),
                    Part.fromText("{\"businessName\":\"Shop\"}"))
                .build());

    assertThat(content.isArray()).isTrue();
    assertThat(content.size()).isEqualTo(3);
    assertThat(content.get(0).get("type").asText()).isEqualTo("text");
    assertThat(content.get(0).get("text").asText()).isEqualTo("Extract brand values.");
    assertThat(content.get(1).get("type").asText()).isEqualTo("image_url");
    assertThat(content.get(1).get("image_url").get("url").asText())
        .isEqualTo("data:image/png;base64," + Base64.getEncoder().encodeToString(PNG));
    assertThat(content.get(2).get("text").asText()).isEqualTo("{\"businessName\":\"Shop\"}");
  }

  @Test
  public void textOnlyUserMessagesKeepPlainStringContent() throws Exception {
    JsonNode content =
        sentUserContent(Content.builder().role("user").parts(Part.fromText("hi")).build());

    assertThat(content.isTextual()).isTrue();
    assertThat(content.asText()).isEqualTo("hi");
  }

  /** Sends one request to a local server and returns the user message's {@code content}. */
  private static JsonNode sentUserContent(Content userMessage) throws Exception {
    AtomicReference<String> requestBody = new AtomicReference<>();
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/",
        exchange -> {
          requestBody.set(
              new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
          byte[] bytes =
              ("{\"id\":\"c1\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"gpt-test\","
                      + "\"choices\":[{\"index\":0,\"finish_reason\":\"stop\","
                      + "\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}")
                  .getBytes(StandardCharsets.UTF_8);
          exchange.getResponseHeaders().add("Content-Type", "application/json");
          exchange.sendResponseHeaders(200, bytes.length);
          try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
          }
        });
    server.start();
    try {
      OpenAILlm llm =
          new OpenAILlm(
              "gpt-test",
              OpenAIOkHttpClient.builder()
                  .apiKey("test")
                  .baseUrl("http://127.0.0.1:" + server.getAddress().getPort())
                  .maxRetries(0)
                  .build());
      llm.generateContent(
              LlmRequest.builder().contents(ImmutableList.of(userMessage)).build(), false)
          .blockingFirst();
    } finally {
      server.stop(0);
    }
    JsonNode messages = new ObjectMapper().readTree(requestBody.get()).get("messages");
    return messages.get(messages.size() - 1).get("content");
  }
}
