/*
 * Copyright 2025 Google LLC
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

package com.google.adk.artifacts;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.genai.types.Part;
import dev.notify.artifact.ArtifactEngine;
import dev.notify.artifact.model.Artifact;
import dev.notify.artifact.model.ArtifactStatus;
import dev.notify.artifact.model.Requests;
import dev.notify.artifact.util.Checksum;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link NotifyArtifactService}. */
@RunWith(JUnit4.class)
public final class NotifyArtifactServiceTest {
  private RecordingEngine engine;
  private NotifyArtifactService service;

  @Before
  public void setUp() {
    engine = new RecordingEngine();
    service = new NotifyArtifactService(engine);
  }

  @Test
  public void saveLoadListAndVersion_useEngineApis() {
    int first =
        service.saveArtifact("app", "user", "session", "notes.txt", bytes("first")).blockingGet();
    int second =
        service.saveArtifact("app", "user", "session", "notes.txt", bytes("second")).blockingGet();

    assertThat(first).isEqualTo(0);
    assertThat(second).isEqualTo(1);
    assertThat(service.listArtifactKeys("app", "user", "session").blockingGet().filenames())
        .containsExactly("notes.txt");
    assertThat(service.listVersions("app", "user", "session", "notes.txt").blockingGet())
        .containsExactly(0, 1)
        .inOrder();
    assertThat(data(load("session", "notes.txt", null))).isEqualTo(utf8("second"));
    assertThat(data(load("session", "notes.txt", 0))).isEqualTo(utf8("first"));
    assertThat(engine.ingestCalls).isEqualTo(2);
    assertThat(engine.contentCalls).isEqualTo(2);
  }

  @Test
  public void userNamespace_isVisibleAcrossSessions() {
    service
        .saveArtifact("app", "user", "session-a", "user:profile.txt", bytes("profile"))
        .blockingGet();

    assertThat(service.listArtifactKeys("app", "user", "session-b").blockingGet().filenames())
        .containsExactly("user:profile.txt");
    assertThat(data(load("session-b", "user:profile.txt", null))).isEqualTo(utf8("profile"));
  }

  @Test
  public void sessionNamespace_isIsolatedByUserAndSession() {
    service
        .saveArtifact("app", "user-a", "session-a", "private.txt", bytes("private"))
        .blockingGet();

    assertThat(service.listArtifactKeys("app", "user-a", "session-b").blockingGet().filenames())
        .isEmpty();
    assertThat(service.listArtifactKeys("app", "user-b", "session-a").blockingGet().filenames())
        .isEmpty();
  }

  @Test
  public void delete_removesEveryLogicalVersionThroughEngine() {
    service.saveArtifact("app", "user", "session", "notes.txt", bytes("first")).blockingGet();
    service.saveArtifact("app", "user", "session", "notes.txt", bytes("second")).blockingGet();

    service.deleteArtifact("app", "user", "session", "notes.txt").blockingAwait();

    assertThat(engine.deleteCalls).isEqualTo(2);
    assertThat(service.listVersions("app", "user", "session", "notes.txt").blockingGet()).isEmpty();
  }

  @Test
  public void customAccessResolver_controlsTenantAndPrincipal() {
    service =
        new NotifyArtifactService(
            engine,
            (appName, userId, sessionId) ->
                new NotifyArtifactService.AccessContext("tenant-42", "principal-42"),
            100,
            1024);

    service.saveArtifact("app", "user", "session", "notes.txt", bytes("content")).blockingGet();

    assertThat(engine.lastTenant).isEqualTo("tenant-42");
    assertThat(engine.lastPrincipal).isEqualTo("principal-42");
  }

  @Test
  public void save_rejectsPartsWithoutInlineData() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            service
                .saveArtifact("app", "user", "session", "notes.txt", Part.builder().build())
                .blockingGet());
  }

  private Part load(String sessionId, String filename, Integer version) {
    Optional<Part> loaded =
        service
            .loadArtifact("app", "user", sessionId, filename, version)
            .map(Optional::of)
            .defaultIfEmpty(Optional.empty())
            .blockingGet();
    return loaded.orElseThrow();
  }

  private static Part bytes(String value) {
    return Part.fromBytes(utf8(value), "text/plain");
  }

  private static byte[] data(Part part) {
    return part.inlineData().orElseThrow().data().orElseThrow();
  }

  private static byte[] utf8(String value) {
    return value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
  }

  private static final class RecordingEngine implements ArtifactEngine {
    private final AtomicInteger ids = new AtomicInteger();
    private final List<Artifact> artifacts = new ArrayList<>();
    private final Map<String, byte[]> contents = new HashMap<>();
    private int ingestCalls;
    private int contentCalls;
    private int deleteCalls;
    private String lastTenant;
    private String lastPrincipal;

    @Override
    public Artifact ingest(Requests.Ingest request) throws IOException {
      record(request.tenantId(), request.principalId());
      ingestCalls++;
      byte[] content = request.content().readAllBytes();
      String id = "artifact-" + ids.incrementAndGet();
      Instant now = Instant.now();
      Artifact artifact =
          new Artifact(
              id,
              request.tenantId(),
              request.idempotencyKey(),
              "fingerprint",
              "UPLOAD",
              null,
              request.originalName(),
              request.declaredMediaType(),
              content.length,
              Checksum.sha256(new ByteArrayInputStream(content)),
              "objects/" + id,
              Path.of("spool", id),
              ArtifactStatus.Storage.STORED,
              ArtifactStatus.Index.READY,
              1,
              request.metadata(),
              null,
              null,
              now,
              now);
      artifacts.add(artifact);
      contents.put(id, content);
      return artifact;
    }

    @Override
    public Artifact metadata(String principalId, String tenantId, String artifactId) {
      record(tenantId, principalId);
      return artifacts.stream()
          .filter(artifact -> artifact.id().equals(artifactId))
          .findFirst()
          .orElseThrow();
    }

    @Override
    public List<Artifact> listMetadata(String principalId, String tenantId, int limit) {
      record(tenantId, principalId);
      return artifacts.stream()
          .filter(artifact -> tenantId.equals(artifact.tenantId()))
          .limit(limit)
          .toList();
    }

    @Override
    public InputStream content(String principalId, String tenantId, String artifactId) {
      record(tenantId, principalId);
      contentCalls++;
      return new ByteArrayInputStream(contents.get(artifactId));
    }

    @Override
    public String extractedText(
        String principalId, String tenantId, String artifactId, int maxCharacters) {
      record(tenantId, principalId);
      return new String(contents.get(artifactId), java.nio.charset.StandardCharsets.UTF_8);
    }

    @Override
    public List<Requests.SearchHit> search(Requests.Search request) {
      record(request.tenantId(), request.principalId());
      return List.of();
    }

    @Override
    public void delete(String principalId, String tenantId, String artifactId) {
      record(tenantId, principalId);
      deleteCalls++;
      artifacts.removeIf(artifact -> artifact.id().equals(artifactId));
      contents.remove(artifactId);
    }

    private void record(String tenantId, String principalId) {
      lastTenant = tenantId;
      lastPrincipal = principalId;
    }
  }
}
