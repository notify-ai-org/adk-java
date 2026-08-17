/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */
package com.google.adk.flows.llmflows;

import static com.google.adk.testing.TestUtils.createInvocationContext;
import static com.google.adk.testing.TestUtils.createLlmResponse;
import static com.google.adk.testing.TestUtils.createTestAgent;
import static com.google.adk.testing.TestUtils.createTestLlm;
import static com.google.common.truth.Truth.assertThat;

import com.google.adk.agents.InvocationContext;
import com.google.adk.agents.RetryConfig;
import com.google.adk.agents.RunConfig;
import com.google.adk.models.LlmCallsLimitExceededException;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.testing.TestLlm;
import com.google.genai.errors.ServerException;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.subscribers.TestSubscriber;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LlmRetryPolicyTest {
  private static final LlmRequest REQUEST = LlmRequest.builder().build();
  private static final LlmResponse RESPONSE =
      createLlmResponse(Content.fromParts(Part.fromText("ok")));

  @Test
  public void execute_retryDisabled_propagatesFirstFailure() {
    AtomicInteger attempts = new AtomicInteger();
    TestLlm llm =
        createTestLlm(
            () -> {
              attempts.incrementAndGet();
              return Flowable.error(new IOException("unavailable"));
            });

    LlmRetryPolicy.execute(context(llm, RetryConfig.disabled(), 10), llm, REQUEST, false)
        .test()
        .assertError(IOException.class)
        .assertNoValues();

    assertThat(attempts.get()).isEqualTo(1);
  }

  @Test
  public void execute_transientFailure_retriesAndSucceeds() {
    AtomicInteger attempts = new AtomicInteger();
    TestLlm llm =
        createTestLlm(
            () ->
                attempts.incrementAndGet() == 1
                    ? Flowable.error(new ServerException(503, "unavailable", "UNAVAILABLE"))
                    : Flowable.just(RESPONSE));

    LlmRetryPolicy.execute(context(llm, retryConfig(3), 10), llm, REQUEST, false)
        .test()
        .awaitDone(5, TimeUnit.SECONDS)
        .assertResult(RESPONSE);

    assertThat(attempts.get()).isEqualTo(2);
  }

  @Test
  public void execute_nonRetryableFailure_doesNotRetry() {
    AtomicInteger attempts = new AtomicInteger();
    TestLlm llm =
        createTestLlm(
            () -> {
              attempts.incrementAndGet();
              return Flowable.error(new ServerException(400, "bad request", "INVALID_ARGUMENT"));
            });

    LlmRetryPolicy.execute(context(llm, retryConfig(3), 10), llm, REQUEST, false)
        .test()
        .assertError(ServerException.class)
        .assertNoValues();

    assertThat(attempts.get()).isEqualTo(1);
  }

  @Test
  public void execute_attemptsExhausted_returnsLastFailure() {
    AtomicInteger attempts = new AtomicInteger();
    TestLlm llm =
        createTestLlm(
            () -> {
              attempts.incrementAndGet();
              return Flowable.error(new IOException("unavailable"));
            });

    LlmRetryPolicy.execute(context(llm, retryConfig(3), 10), llm, REQUEST, false)
        .test()
        .awaitDone(5, TimeUnit.SECONDS)
        .assertError(IOException.class)
        .assertNoValues();

    assertThat(attempts.get()).isEqualTo(3);
  }

  @Test
  public void execute_retriesConsumeLlmCallBudget() {
    AtomicInteger attempts = new AtomicInteger();
    TestLlm llm =
        createTestLlm(
            () -> {
              attempts.incrementAndGet();
              return Flowable.error(new IOException("unavailable"));
            });

    LlmRetryPolicy.execute(context(llm, retryConfig(3), 2), llm, REQUEST, false)
        .test()
        .awaitDone(5, TimeUnit.SECONDS)
        .assertError(LlmCallsLimitExceededException.class)
        .assertNoValues();

    assertThat(attempts.get()).isEqualTo(2);
  }

  @Test
  public void execute_streamFailsAfterResponse_doesNotRetry() {
    AtomicInteger attempts = new AtomicInteger();
    TestLlm llm =
        createTestLlm(
            () -> {
              attempts.incrementAndGet();
              return Flowable.concat(
                  Flowable.just(RESPONSE), Flowable.error(new IOException("stream interrupted")));
            });

    TestSubscriber<LlmResponse> subscriber =
        LlmRetryPolicy.execute(context(llm, retryConfig(3), 10), llm, REQUEST, true).test();
    subscriber.assertValue(RESPONSE).assertError(IOException.class);

    assertThat(attempts.get()).isEqualTo(1);
  }

  @Test
  public void runConfig_copyPreservesRetryConfig() {
    RetryConfig retryConfig = retryConfig(4);
    RunConfig copied =
        RunConfig.builder(RunConfig.builder().retryConfig(retryConfig).build()).build();

    assertThat(copied.retryConfig()).isEqualTo(retryConfig);
  }

  private static InvocationContext context(TestLlm llm, RetryConfig retryConfig, int maxLlmCalls) {
    RunConfig runConfig =
        RunConfig.builder().retryConfig(retryConfig).maxLlmCalls(maxLlmCalls).build();
    return createInvocationContext(createTestAgent(llm), runConfig);
  }

  private static RetryConfig retryConfig(int maxAttempts) {
    return RetryConfig.builder()
        .maxAttempts(maxAttempts)
        .initialBackoff(Duration.ZERO)
        .maxBackoff(Duration.ZERO)
        .jitterRatio(0.0)
        .build();
  }
}
