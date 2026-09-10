/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.adk.models;

import static com.google.common.truth.Truth.assertThat;

import com.google.genai.types.FunctionCallingConfig;
import com.google.genai.types.FunctionCallingConfigMode.Known;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.ToolConfig;
import com.openai.models.chat.completions.ChatCompletionToolChoiceOption;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class OpenAILlmToolChoiceTest {

  @Test
  public void toolChoice_anyRequiresToolCall() {
    LlmRequest request = requestWithMode(Known.ANY);

    assertThat(OpenAILlm.toolChoice(request).asAuto())
        .isEqualTo(ChatCompletionToolChoiceOption.Auto.REQUIRED);
  }

  @Test
  public void toolChoice_noneDisablesToolCall() {
    LlmRequest request = requestWithMode(Known.NONE);

    assertThat(OpenAILlm.toolChoice(request).asAuto())
        .isEqualTo(ChatCompletionToolChoiceOption.Auto.NONE);
  }

  private LlmRequest requestWithMode(Known mode) {
    return LlmRequest.builder()
        .config(
            GenerateContentConfig.builder()
                .toolConfig(
                    ToolConfig.builder()
                        .functionCallingConfig(FunctionCallingConfig.builder().mode(mode).build())
                        .build())
                .build())
        .build();
  }
}
