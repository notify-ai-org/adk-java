package com.google.adk.models;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import io.reactivex.rxjava3.core.Flowable;
import java.util.Objects;
import java.util.Set;

/** OpenAI-compatible LLM provider for Groq chat completion models. */
public final class GroqLlm extends BaseLlm {

  private static final String GROQ_MODEL_PREFIX = "groq/";
  private static final String DEFAULT_BASE_URL = "https://api.groq.com/openai/v1";
  private final OpenAILlm delegate;

  private GroqLlm(String model, OpenAIClient client) {
    super(model);
    this.delegate = new OpenAILlm(stripGroqPrefix(model), client);
  }

  private GroqLlm(String model, OpenAIClient client, int maxTokens) {
    super(model);
    this.delegate = new OpenAILlm(stripGroqPrefix(model), client, maxTokens);
  }

  @Override
  public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
    return delegate.generateContent(
        llmRequest.toBuilder().model(stripGroqPrefix(model())).build(), stream);
  }

  @Override
  public boolean isExceptionRetryable(Throwable exception, Set<Integer> retryableStatusCodes) {
    return delegate.isExceptionRetryable(exception, retryableStatusCodes);
  }

  @Override
  public BaseLlmConnection connect(LlmRequest llmRequest) {
    throw new UnsupportedOperationException("Live connection is not supported for Groq models.");
  }

  private static String stripGroqPrefix(String modelName) {
    if (modelName != null && modelName.startsWith(GROQ_MODEL_PREFIX)) {
      return modelName.substring(GROQ_MODEL_PREFIX.length());
    }
    return modelName;
  }

  private static OpenAIClient defaultClient() {
    String apiKey = System.getenv("GROQ_API_KEY");
    if (apiKey == null || apiKey.isBlank()) {
      throw new IllegalStateException("GROQ_API_KEY must be set for groq/* models.");
    }
    return OpenAIOkHttpClient.builder()
        .apiKey(apiKey)
        .baseUrl(envOrDefault("GROQ_API_BASE_URL", DEFAULT_BASE_URL))
        .build();
  }

  private static String envOrDefault(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isBlank() ? defaultValue : value;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private String modelName;
    private OpenAIClient apiClient;
    private int maxTokens;

    private Builder() {}

    @CanIgnoreReturnValue
    public Builder modelName(String modelName) {
      this.modelName = modelName;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder apiClient(OpenAIClient apiClient) {
      this.apiClient = apiClient;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder maxTokens(int maxTokens) {
      this.maxTokens = maxTokens;
      return this;
    }

    public GroqLlm build() {
      Objects.requireNonNull(modelName, "modelName must be set.");
      OpenAIClient client = apiClient != null ? apiClient : defaultClient();
      if (maxTokens > 0) {
        return new GroqLlm(modelName, client, maxTokens);
      }
      return new GroqLlm(modelName, client);
    }
  }
}
