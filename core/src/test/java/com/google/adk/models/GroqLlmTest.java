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
}
