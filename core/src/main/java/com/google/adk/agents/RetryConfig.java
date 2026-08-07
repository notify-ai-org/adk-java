/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */
package com.google.adk.agents;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;

/** Opt-in retry configuration for transient LLM provider failures. */
@AutoValue
public abstract class RetryConfig {
  public abstract int maxAttempts();

  public abstract Duration initialBackoff();

  public abstract Duration maxBackoff();

  public abstract double multiplier();

  public abstract double jitterRatio();

  public abstract ImmutableSet<Integer> retryableStatusCodes();

  /** Returns a retry policy that performs exactly one provider attempt. */
  public static RetryConfig disabled() {
    return builder().build();
  }

  public static Builder builder() {
    return new AutoValue_RetryConfig.Builder()
        .maxAttempts(1)
        .initialBackoff(Duration.ofMillis(500))
        .maxBackoff(Duration.ofSeconds(8))
        .multiplier(2.0)
        .jitterRatio(0.2)
        .retryableStatusCodes(ImmutableSet.of(408, 429, 500, 502, 503, 504));
  }

  /** Builder for {@link RetryConfig}. */
  @AutoValue.Builder
  public abstract static class Builder {
    @CanIgnoreReturnValue
    public abstract Builder maxAttempts(int maxAttempts);

    @CanIgnoreReturnValue
    public abstract Builder initialBackoff(Duration initialBackoff);

    @CanIgnoreReturnValue
    public abstract Builder maxBackoff(Duration maxBackoff);

    @CanIgnoreReturnValue
    public abstract Builder multiplier(double multiplier);

    @CanIgnoreReturnValue
    public abstract Builder jitterRatio(double jitterRatio);

    @CanIgnoreReturnValue
    public abstract Builder retryableStatusCodes(Iterable<Integer> retryableStatusCodes);

    abstract RetryConfig autoBuild();

    public RetryConfig build() {
      RetryConfig config = autoBuild();
      if (config.maxAttempts() < 1) {
        throw new IllegalArgumentException("maxAttempts must be at least 1");
      }
      if (config.initialBackoff().isNegative() || config.maxBackoff().isNegative()) {
        throw new IllegalArgumentException("retry backoff durations must not be negative");
      }
      if (config.initialBackoff().compareTo(config.maxBackoff()) > 0) {
        throw new IllegalArgumentException("initialBackoff must not exceed maxBackoff");
      }
      if (!Double.isFinite(config.multiplier()) || config.multiplier() < 1.0) {
        throw new IllegalArgumentException("multiplier must be finite and at least 1.0");
      }
      if (!Double.isFinite(config.jitterRatio())
          || config.jitterRatio() < 0.0
          || config.jitterRatio() > 1.0) {
        throw new IllegalArgumentException("jitterRatio must be between 0.0 and 1.0");
      }
      return config;
    }
  }
}
