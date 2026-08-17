package com.google.adk.models;

import com.fasterxml.jackson.core.type.TypeReference;
import com.google.adk.JsonBaseModel;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.JsonValue;
import com.openai.errors.OpenAIIoException;
import com.openai.errors.OpenAIRetryableException;
import com.openai.errors.OpenAIServiceException;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionAssistantMessageParam;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessage;
import com.openai.models.chat.completions.ChatCompletionMessageFunctionToolCall;
import com.openai.models.chat.completions.ChatCompletionMessageParam;
import com.openai.models.chat.completions.ChatCompletionMessageToolCall;
import com.openai.models.chat.completions.ChatCompletionSystemMessageParam;
import com.openai.models.chat.completions.ChatCompletionTool;
import com.openai.models.chat.completions.ChatCompletionToolChoiceOption;
import com.openai.models.chat.completions.ChatCompletionToolMessageParam;
import com.openai.models.chat.completions.ChatCompletionUserMessageParam;
import io.reactivex.rxjava3.core.Flowable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;
import org.jspecify.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents the OpenAI Generative AI model.
 *
 * <p>This class provides methods for interacting with OpenAI models via the Chat Completions API.
 * Streaming and live connections are not currently supported.
 */
public class OpenAILlm extends BaseLlm {

  private static final Logger logger = LoggerFactory.getLogger(OpenAILlm.class);
  private int maxTokens = 8192;
  private final OpenAIClient client;

  /**
   * Constructs a new OpenAILlm instance.
   *
   * @param model The name of the OpenAI model to use (e.g., "gpt-4o").
   * @param client The OpenAI API client instance.
   */
  public OpenAILlm(String model, OpenAIClient client) {
    super(model);
    this.client = client;
  }

  public OpenAILlm(String model, OpenAIClient client, int maxTokens) {
    super(model);
    this.client = client;
    this.maxTokens = maxTokens;
  }

  public OpenAILlm(String model) {
    super(model);
    this.maxTokens = 8192;
    OpenAIClient client = OpenAIOkHttpClient.fromEnv();
    this.client = client;
  }

  @Override
  public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
    // Build the list of messages from the ADK Content objects.
    List<ChatCompletionMessageParam> messages = new ArrayList<>();

    // Extract system instruction and add as a system message.
    String systemText = "";
    Optional<GenerateContentConfig> configOpt = llmRequest.config();
    if (configOpt.isPresent()) {
      Optional<Content> systemInstructionOpt = configOpt.get().systemInstruction();
      if (systemInstructionOpt.isPresent()) {
        String extractedSystemText =
            systemInstructionOpt.get().parts().orElse(ImmutableList.of()).stream()
                .filter(p -> p.text().isPresent())
                .map(p -> p.text().get())
                .collect(Collectors.joining("\n"));
        if (!extractedSystemText.isEmpty()) {
          systemText = extractedSystemText;
        }
      }
    }

    if (!systemText.isEmpty()) {
      messages.add(
          ChatCompletionMessageParam.ofSystem(
              ChatCompletionSystemMessageParam.builder().content(systemText).build()));
    }

    // Convert each ADK Content to OpenAI ChatCompletionMessageParam entries.
    // A single ADK content may contain multiple function responses; OpenAI requires
    // one tool message per tool_call_id.
    Set<String> expectedToolResponseIds = new HashSet<>();
    List<Content> contents = llmRequest.contents();
    for (int i = 0; i < contents.size(); i++) {
      Content content = contents.get(i);
      if (hasFunctionCall(content)) {
        Set<String> callIds = functionCallIds(content);
        if (!hasImmediateFunctionResponsesForAll(contents, i, callIds)) {
          logger.warn("Skipping orphan OpenAI tool-call history entry with ids={}", callIds);
          continue;
        }
        messages.addAll(contentToOpenAIMessageParams(content));
        expectedToolResponseIds.addAll(callIds);
        continue;
      }

      if (hasFunctionResponse(content)) {
        if (expectedToolResponseIds.isEmpty()) {
          logger.warn(
              "Skipping orphan OpenAI tool-response history entry with ids={}",
              functionResponseIds(content));
          continue;
        }
        messages.addAll(functionResponseMessages(content, expectedToolResponseIds));
        expectedToolResponseIds.removeAll(functionResponseIds(content));
        continue;
      }

      messages.addAll(contentToOpenAIMessageParams(content));
    }

    // Convert ADK function declarations to OpenAI tools.
    List<ChatCompletionTool> tools = ImmutableList.of();
    if (llmRequest.config().isPresent()
        && llmRequest.config().get().tools().isPresent()
        && !llmRequest.config().get().tools().get().isEmpty()
        && llmRequest.config().get().tools().get().get(0).functionDeclarations().isPresent()) {
      tools =
          llmRequest.config().get().tools().get().get(0).functionDeclarations().get().stream()
              .map(this::functionDeclarationToOpenAITool)
              .collect(Collectors.toList());
    }

    // Build the request params.
    ChatCompletionCreateParams.Builder paramsBuilder =
        ChatCompletionCreateParams.builder()
            .model(llmRequest.model().orElse(model()))
            .messages(messages)
            .maxCompletionTokens((long) this.maxTokens);

    if (!tools.isEmpty()) {
      paramsBuilder.tools(tools);
      paramsBuilder.toolChoice(ChatCompletionToolChoiceOption.Auto.AUTO);
      paramsBuilder.parallelToolCalls(false);
    }

    ChatCompletion completion = client.chat().completions().create(paramsBuilder.build());

    logger.debug("OpenAI response: {}", completion);

    return Flowable.just(convertOpenAIResponseToLlmResponse(completion));
  }

  private ChatCompletionMessageParam contentToOpenAIMessageParam(Content content) {
    List<ChatCompletionMessageParam> messages = contentToOpenAIMessageParams(content);
    return messages.isEmpty() ? null : messages.get(0);
  }

  private List<ChatCompletionMessageParam> contentToOpenAIMessageParams(Content content) {
    String role = content.role().orElse("");
    List<Part> parts = content.parts().orElse(ImmutableList.of());

    if (hasFunctionCall(content)) {
      // Build an assistant message with tool calls.
      String textContent =
          parts.stream()
              .filter(p -> p.text().isPresent())
              .map(p -> p.text().get())
              .collect(Collectors.joining("\n"));

      List<ChatCompletionMessageToolCall> toolCalls =
          parts.stream()
              .filter(p -> p.functionCall().isPresent())
              .map(
                  p -> {
                    FunctionCall fc = p.functionCall().get();
                    String argsJson = serializeToJson(fc.args().orElse(ImmutableMap.of()));
                    return ChatCompletionMessageToolCall.ofFunction(
                        ChatCompletionMessageFunctionToolCall.builder()
                            .id(fc.id().orElse(""))
                            .function(
                                ChatCompletionMessageFunctionToolCall.Function.builder()
                                    .name(fc.name().orElseThrow())
                                    .arguments(argsJson)
                                    .build())
                            .build());
                  })
              .collect(Collectors.toList());

      ChatCompletionAssistantMessageParam.Builder assistantBuilder =
          ChatCompletionAssistantMessageParam.builder().toolCalls(toolCalls);

      if (!textContent.isEmpty()) {
        assistantBuilder.content(textContent);
      }

      return List.of(ChatCompletionMessageParam.ofAssistant(assistantBuilder.build()));
    } else if (hasFunctionResponse(content)) {
      return functionResponseMessages(content, null);
    }

    // Regular text message: determine role.
    String textContent =
        parts.stream()
            .filter(p -> p.text().isPresent())
            .map(p -> p.text().get())
            .collect(Collectors.joining("\n"));

    if (role.equals("model") || role.equals("assistant")) {
      return List.of(
          ChatCompletionMessageParam.ofAssistant(
              ChatCompletionAssistantMessageParam.builder().content(textContent).build()));
    } else {
      return List.of(
          ChatCompletionMessageParam.ofUser(
              ChatCompletionUserMessageParam.builder().content(textContent).build()));
    }
  }

  private boolean hasFunctionCall(Content content) {
    return content.parts().orElse(ImmutableList.of()).stream()
        .anyMatch(p -> p.functionCall().isPresent());
  }

  private boolean hasFunctionResponse(Content content) {
    return content.parts().orElse(ImmutableList.of()).stream()
        .anyMatch(p -> p.functionResponse().isPresent());
  }

  private Set<String> functionCallIds(Content content) {
    return content.parts().orElse(ImmutableList.of()).stream()
        .flatMap(part -> part.functionCall().stream())
        .flatMap(call -> call.id().stream())
        .filter(id -> !id.isBlank())
        .collect(Collectors.toSet());
  }

  private Set<String> functionResponseIds(Content content) {
    return content.parts().orElse(ImmutableList.of()).stream()
        .flatMap(part -> part.functionResponse().stream())
        .flatMap(response -> response.id().stream())
        .filter(id -> !id.isBlank())
        .collect(Collectors.toSet());
  }

  private boolean hasImmediateFunctionResponsesForAll(
      List<Content> contents, int functionCallIndex, Set<String> requiredIds) {
    if (requiredIds.isEmpty()) {
      return false;
    }

    Set<String> remaining = new HashSet<>(requiredIds);
    for (int i = functionCallIndex + 1; i < contents.size(); i++) {
      Content next = contents.get(i);
      if (!hasFunctionResponse(next)) {
        break;
      }
      remaining.removeAll(functionResponseIds(next));
      if (remaining.isEmpty()) {
        return true;
      }
    }
    return false;
  }

  private List<ChatCompletionMessageParam> functionResponseMessages(
      Content content, @Nullable Set<String> allowedResponseIds) {
    List<ChatCompletionMessageParam> toolMessages = new ArrayList<>();
    for (Part p : content.parts().orElse(ImmutableList.of())) {
      if (p.functionResponse().isEmpty()) {
        continue;
      }
      String responseId = p.functionResponse().get().id().orElse("");
      if (allowedResponseIds != null && !allowedResponseIds.contains(responseId)) {
        continue;
      }
      String responseContent = "";
      if (p.functionResponse().get().response().isPresent()) {
        Map<String, Object> responseData = p.functionResponse().get().response().get();
        Object resultObj = responseData.get("result");
        if (resultObj != null) {
          responseContent = resultObj.toString();
        } else {
          responseContent = serializeToJson(responseData);
        }
      }
      toolMessages.add(
          ChatCompletionMessageParam.ofTool(
              ChatCompletionToolMessageParam.builder()
                  .toolCallId(responseId)
                  .content(responseContent)
                  .build()));
    }
    return toolMessages;
  }

  private String serializeToJson(Object obj) {
    try {
      return JsonBaseModel.getMapper().writeValueAsString(obj);
    } catch (Exception e) {
      logger.warn("Failed to serialize object to JSON", e);
      return String.valueOf(obj);
    }
  }

  private void updateTypeString(Map<String, Object> valueDict) {
    if (valueDict == null) {
      return;
    }
    if (valueDict.containsKey("type")) {
      valueDict.put("type", ((String) valueDict.get("type")).toLowerCase());
    }

    if (valueDict.containsKey("items")) {
      updateTypeString((Map<String, Object>) valueDict.get("items"));

      if (valueDict.get("items") instanceof Map
          && ((Map) valueDict.get("items")).containsKey("properties")) {
        Map<String, Object> properties =
            (Map<String, Object>) ((Map) valueDict.get("items")).get("properties");
        if (properties != null) {
          for (Object value : properties.values()) {
            if (value instanceof Map) {
              updateTypeString((Map<String, Object>) value);
            }
          }
        }
      }
    }
  }

  private ChatCompletionTool functionDeclarationToOpenAITool(
      FunctionDeclaration functionDeclaration) {
    Map<String, Map<String, Object>> properties = new HashMap<>();
    if (functionDeclaration.parameters().isPresent()
        && functionDeclaration.parameters().get().properties().isPresent()) {
      functionDeclaration
          .parameters()
          .get()
          .properties()
          .get()
          .forEach(
              (key, schema) -> {
                Map<String, Object> schemaMap =
                    JsonBaseModel.getMapper()
                        .convertValue(schema, new TypeReference<Map<String, Object>>() {});
                updateTypeString(schemaMap);
                properties.put(key, schemaMap);
              });
    }

    FunctionParameters.Builder paramsBuilder = FunctionParameters.builder();
    paramsBuilder.putAdditionalProperty("type", JsonValue.from("object"));
    paramsBuilder.putAdditionalProperty("properties", JsonValue.from(properties));

    return ChatCompletionTool.ofFunction(
        com.openai.models.chat.completions.ChatCompletionFunctionTool.builder()
            .function(
                FunctionDefinition.builder()
                    .name(functionDeclaration.name().orElseThrow())
                    .description(functionDeclaration.description().orElse(""))
                    .parameters(paramsBuilder.build())
                    .build())
            .build());
  }

  private LlmResponse convertOpenAIResponseToLlmResponse(ChatCompletion completion) {
    LlmResponse.Builder responseBuilder = LlmResponse.builder();
    List<Part> parts = new ArrayList<>();

    if (completion.choices() != null && !completion.choices().isEmpty()) {
      ChatCompletionMessage message = completion.choices().get(0).message();

      // Handle text content.
      if (message.content().isPresent()) {
        parts.add(Part.builder().text(message.content().get()).build());
      }

      // Handle tool calls.
      if (message.toolCalls().isPresent()) {
        for (ChatCompletionMessageToolCall toolCall : message.toolCalls().get()) {
          if (toolCall.isFunction()) {
            ChatCompletionMessageFunctionToolCall functionToolCall = toolCall.asFunction();
            Map<String, Object> args;
            try {
              args =
                  JsonBaseModel.getMapper()
                      .readValue(
                          functionToolCall.function().arguments(),
                          new TypeReference<Map<String, Object>>() {});
            } catch (Exception e) {
              logger.warn("Failed to parse function arguments as JSON", e);
              args = ImmutableMap.of();
            }
            parts.add(
                Part.builder()
                    .functionCall(
                        FunctionCall.builder()
                            .id(functionToolCall.id())
                            .name(functionToolCall.function().name())
                            .args(args)
                            .build())
                    .build());
          }
        }
      }

      if (!parts.isEmpty()) {
        responseBuilder.content(
            Content.builder().role("model").parts(ImmutableList.copyOf(parts)).build());
      }
    }
    return responseBuilder.build();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private String modelName;
    private OpenAIClient apiClient;
    private int maxTokens;

    private Builder() {}

    /**
     * Sets the name of the OpenAI model to use.
     *
     * @param modelName The model name (e.g., "gpt-4o").
     * @return This builder.
     */
    @CanIgnoreReturnValue
    public Builder modelName(String modelName) {
      this.modelName = modelName;
      return this;
    }

    /**
     * Sets the explicit {@link com.op} instance for making API calls. If this is set, apiKey and
     * vertexCredentials will be ignored.
     *
     * @param apiClient The client instance.
     * @return This builder.
     */
    @CanIgnoreReturnValue
    public Builder apiClient(OpenAIClient apiClient) {
      this.apiClient = apiClient;
      return this;
    }

    /**
     * Sets the maximum number of tokens to generate. If {@link #apiClient(Client)} is also set, the
     * explicit client will take precedence. If {@link #vertexCredentials(VertexCredentials)} is
     * also set, this apiKey will take precedence.
     *
     * @param maxTokens The maximum number of tokens to generate.
     * @return This builder.
     */
    @CanIgnoreReturnValue
    public Builder maxTokens(int maxTokens) {
      this.maxTokens = maxTokens;
      return this;
    }

    /**
     * Builds the {@link Gemini} instance.
     *
     * @return A new {@link Gemini} instance.
     * @throws NullPointerException if modelName is null.
     */
    public OpenAILlm build() {
      Objects.requireNonNull(modelName, "modelName must be set.");
      if (apiClient != null) {
        return new OpenAILlm(modelName, apiClient);
      } else {
        return new OpenAILlm(modelName, OpenAIOkHttpClient.fromEnv());
      }
    }
  }

  @Override
  public BaseLlmConnection connect(LlmRequest llmRequest) {
    throw new UnsupportedOperationException("Live connection is not supported for OpenAI models.");
  }

  @Override
  public boolean isExceptionRetryable(Throwable exception, Set<Integer> retryableStatusCodes) {
    return exception instanceof OpenAIServiceException serviceException
            && retryableStatusCodes.contains(serviceException.statusCode())
        || exception instanceof OpenAIIoException
        || exception instanceof OpenAIRetryableException
        || exception instanceof IOException
        || exception instanceof TimeoutException;
  }
}
