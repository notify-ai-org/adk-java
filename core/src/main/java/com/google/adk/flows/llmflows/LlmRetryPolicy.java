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

import com.google.adk.agents.InvocationContext;
import com.google.adk.agents.RetryConfig;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.LlmCallsLimitExceededException;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import io.opentelemetry.api.trace.Span;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Executes an LLM provider call with bounded retries for transient failures. */
public final class LlmRetryPolicy {
  private static final Logger logger = LoggerFactory.getLogger(LlmRetryPolicy.class);

  private LlmRetryPolicy() {}

  public static Flowable<LlmResponse> execute(
      InvocationContext context, BaseLlm llm, LlmRequest request, boolean stream) {
    return executeAttempt(context, llm, request, stream, context.runConfig().retryConfig(), 1);
  }

  private static Flowable<LlmResponse> executeAttempt(
      InvocationContext context,
      BaseLlm llm,
      LlmRequest request,
      boolean stream,
      RetryConfig config,
      int attempt) {
    return Flowable.defer(
        () -> {
          try {
            context.incrementLlmCallsCount();
          } catch (LlmCallsLimitExceededException error) {
            return Flowable.error(error);
          }

          AtomicBoolean emittedResponse = new AtomicBoolean(false);
          return llm.generateContent(request, stream)
              .doOnNext(unused -> emittedResponse.set(true))
              .onErrorResumeNext(
                  error -> {
                    if (emittedResponse.get()
                        || attempt >= config.maxAttempts()
                        || !isRetryable(llm, error, config)) {
                      return Flowable.error(error);
                    }

                    long delayMillis = retryDelayMillis(config, attempt);
                    logger.warn(
                        "Retrying transient LLM failure: invocationId={}, agent={}, model={}, "
                            + "attempt={}/{}, delayMs={}, error={}",
                        context.invocationId(),
                        context.agent().name(),
                        llm.model(),
                        attempt + 1,
                        config.maxAttempts(),
                        delayMillis,
                        error.getClass().getSimpleName());
                    Span.current().setAttribute("gen_ai.retry.attempt", attempt + 1L);
                    Span.current().setAttribute("gen_ai.retry.delay_ms", delayMillis);
                    Span.current()
                        .setAttribute("gen_ai.retry.error_type", error.getClass().getName());

                    return Completable.timer(delayMillis, TimeUnit.MILLISECONDS)
                        .andThen(
                            executeAttempt(context, llm, request, stream, config, attempt + 1));
                  });
        });
  }

  static boolean isRetryable(BaseLlm llm, Throwable error, RetryConfig config) {
    Set<Throwable> visited = Collections.newSetFromMap(new IdentityHashMap<>());
    for (Throwable current = error;
        current != null && visited.add(current);
        current = current.getCause()) {
      if (current instanceof LlmCallsLimitExceededException) {
        return false;
      }
      if (llm.isExceptionRetryable(current, config.retryableStatusCodes())) {
        return true;
      }
    }
    return false;
  }

  private static long retryDelayMillis(RetryConfig config, int failedAttempt) {
    double exponential =
        config.initialBackoff().toMillis()
            * Math.pow(config.multiplier(), Math.max(0, failedAttempt - 1));
    long capped = (long) Math.min(exponential, config.maxBackoff().toMillis());
    if (capped == 0 || config.jitterRatio() == 0.0) {
      return capped;
    }
    double jitter = capped * config.jitterRatio();
    double randomized = ThreadLocalRandom.current().nextDouble(capped - jitter, capped + jitter);
    return Math.max(0L, Math.round(randomized));
  }
}
