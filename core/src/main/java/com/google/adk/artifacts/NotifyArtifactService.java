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

import com.google.common.collect.ImmutableList;
import com.google.genai.types.Blob;
import com.google.genai.types.Part;
import dev.notify.artifact.ArtifactEngine;
import dev.notify.artifact.exception.IdempotencyConflictException;
import dev.notify.artifact.model.Artifact;
import dev.notify.artifact.model.Requests;
import dev.notify.artifact.util.Checksum;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Single;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeSet;
import org.jspecify.annotations.Nullable;

/**
 * ADK artifact service backed by the Notify artifact engine.
 *
 * <p>ADK addresses artifacts by app, user, session, filename, and zero-based version. The engine
 * stores immutable artifacts by ID. This adapter persists the ADK coordinates in engine metadata
 * and resolves them through the engine's authorized, tenant-scoped metadata API. Content,
 * ingestion, metadata, text extraction, RAG search, and deletion never bypass the engine facade.
 *
 * <p>The default access resolver treats the ADK app as the tenant and the ADK user as the
 * principal. Deployments with a different trust model should provide an {@link AccessResolver} that
 * derives both values from trusted application context.
 */
public final class NotifyArtifactService implements BaseArtifactService {
  static final String META_KIND = "notify.adk.kind";
  static final String META_APP = "notify.adk.app";
  static final String META_USER = "notify.adk.user";
  static final String META_SCOPE = "notify.adk.scope";
  static final String META_SESSION = "notify.adk.session";
  static final String META_FILENAME = "notify.adk.filename";
  static final String META_VERSION = "notify.adk.version";

  private static final String ARTIFACT_KIND = "artifact";
  private static final String USER_SCOPE = "user";
  private static final String SESSION_SCOPE = "session";
  private static final String USER_NAMESPACE_PREFIX = "user:";
  private static final String DEFAULT_MEDIA_TYPE = "application/octet-stream";
  private static final int DEFAULT_CATALOG_LIMIT = 10_000;
  private static final int DEFAULT_MAX_INLINE_BYTES = 64 * 1024 * 1024;
  private static final int MAX_SAVE_ATTEMPTS = 4;

  private final ArtifactEngine engine;
  private final AccessResolver accessResolver;
  private final int catalogLimit;
  private final int maxInlineBytes;

  public NotifyArtifactService(ArtifactEngine engine) {
    this(
        engine,
        (appName, userId, sessionId) -> new AccessContext(appName, userId),
        DEFAULT_CATALOG_LIMIT,
        DEFAULT_MAX_INLINE_BYTES);
  }

  public NotifyArtifactService(
      ArtifactEngine engine, AccessResolver accessResolver, int catalogLimit, int maxInlineBytes) {
    this.engine = Objects.requireNonNull(engine, "engine");
    this.accessResolver = Objects.requireNonNull(accessResolver, "accessResolver");
    if (catalogLimit < 1 || catalogLimit > DEFAULT_CATALOG_LIMIT) {
      throw new IllegalArgumentException("catalogLimit must be between 1 and 10000");
    }
    if (maxInlineBytes < 1 || maxInlineBytes == Integer.MAX_VALUE) {
      throw new IllegalArgumentException("maxInlineBytes must be positive and bounded");
    }
    this.catalogLimit = catalogLimit;
    this.maxInlineBytes = maxInlineBytes;
  }

  @Override
  public Single<Integer> saveArtifact(
      String appName, String userId, String sessionId, String filename, Part artifact) {
    return Single.fromCallable(
        () -> save(appName, userId, sessionId, requireFilename(filename), artifact));
  }

  @Override
  public Maybe<Part> loadArtifact(
      String appName, String userId, String sessionId, String filename, @Nullable Integer version) {
    return Maybe.fromCallable(
        () -> {
          String checkedFilename = requireFilename(filename);
          AccessContext access = access(appName, userId, sessionId);
          Optional<Artifact> reference =
              findArtifact(access, appName, userId, sessionId, checkedFilename, version);
          if (reference.isEmpty()) {
            return null;
          }
          return readPart(access, reference.get());
        });
  }

  @Override
  public Single<ListArtifactsResponse> listArtifactKeys(
      String appName, String userId, String sessionId) {
    return Single.fromCallable(
        () -> {
          AccessContext access = access(appName, userId, sessionId);
          TreeSet<String> filenames = new TreeSet<>();
          visibleArtifacts(access, appName, userId, sessionId).stream()
              .map(artifact -> artifact.metadata().get(META_FILENAME))
              .filter(Objects::nonNull)
              .forEach(filenames::add);
          return ListArtifactsResponse.builder().filenames(List.copyOf(filenames)).build();
        });
  }

  @Override
  public Completable deleteArtifact(
      String appName, String userId, String sessionId, String filename) {
    return Completable.fromAction(
        () -> {
          String checkedFilename = requireFilename(filename);
          AccessContext access = access(appName, userId, sessionId);
          List<Artifact> artifacts =
              matchingArtifacts(access, appName, userId, sessionId, checkedFilename);
          for (String artifactId : artifacts.stream().map(Artifact::id).distinct().toList()) {
            try {
              engine.delete(access.principalId(), access.tenantId(), artifactId);
            } catch (NoSuchElementException ignored) {
              // A concurrent or retried delete has already reached the desired state.
            }
          }
        });
  }

  @Override
  public Single<ImmutableList<Integer>> listVersions(
      String appName, String userId, String sessionId, String filename) {
    return Single.fromCallable(
        () -> {
          String checkedFilename = requireFilename(filename);
          AccessContext access = access(appName, userId, sessionId);
          return matchingArtifacts(access, appName, userId, sessionId, checkedFilename).stream()
              .map(NotifyArtifactService::versionOf)
              .flatMap(Optional::stream)
              .distinct()
              .sorted()
              .collect(ImmutableList.toImmutableList());
        });
  }

  @Override
  public Single<Part> saveAndReloadArtifact(
      String appName, String userId, String sessionId, String filename, Part artifact) {
    return saveArtifact(appName, userId, sessionId, filename, artifact)
        .flatMap(
            version ->
                loadArtifact(appName, userId, sessionId, filename, version)
                    .switchIfEmpty(
                        Single.error(
                            new NoSuchElementException(
                                "Saved artifact could not be reloaded: " + filename))));
  }

  /** Returns extracted text for the selected artifact version. */
  public Maybe<String> loadExtractedText(
      String appName,
      String userId,
      String sessionId,
      String filename,
      @Nullable Integer version,
      int maxCharacters) {
    return Maybe.fromCallable(
        () -> {
          AccessContext access = access(appName, userId, sessionId);
          Optional<Artifact> reference =
              findArtifact(access, appName, userId, sessionId, requireFilename(filename), version);
          return reference
              .map(
                  artifact ->
                      engine.extractedText(
                          access.principalId(), access.tenantId(), artifact.id(), maxCharacters))
              .orElse(null);
        });
  }

  /** Searches RAG chunks and removes results outside the caller's ADK user/session scope. */
  public Single<ImmutableList<Requests.SearchHit>> searchArtifacts(
      String appName, String userId, String sessionId, String query, int limit) {
    return Single.fromCallable(
        () -> {
          AccessContext access = access(appName, userId, sessionId);
          int boundedLimit = Math.max(1, Math.min(limit, 100));
          int candidateLimit = Math.min(100, Math.max(boundedLimit, boundedLimit * 4));
          return engine
              .search(
                  new Requests.Search(
                      access.tenantId(),
                      access.principalId(),
                      query,
                      candidateLimit,
                      List.of(),
                      List.of(),
                      null))
              .stream()
              .filter(hit -> isVisible(hit.artifact(), appName, userId, sessionId))
              .limit(boundedLimit)
              .collect(ImmutableList.toImmutableList());
        });
  }

  private int save(String appName, String userId, String sessionId, String filename, Part artifact)
      throws IOException {
    Objects.requireNonNull(artifact, "artifact");
    Blob inlineData =
        artifact
            .inlineData()
            .orElseThrow(() -> new IllegalArgumentException("Artifact must contain inline data"));
    byte[] data =
        inlineData
            .data()
            .orElseThrow(() -> new IllegalArgumentException("Artifact data is required"));
    if (data.length > maxInlineBytes) {
      throw new IllegalArgumentException("Artifact exceeds the configured inline byte limit");
    }
    String mediaType = inlineData.mimeType().orElse(DEFAULT_MEDIA_TYPE);
    AccessContext access = access(appName, userId, sessionId);

    for (int attempt = 0; attempt < MAX_SAVE_ATTEMPTS; attempt++) {
      int version = nextVersion(access, appName, userId, sessionId, filename);
      Map<String, String> metadata =
          referenceMetadata(appName, userId, sessionId, filename, version);
      Requests.Ingest request =
          new Requests.Ingest(
              access.tenantId(),
              access.principalId(),
              idempotencyKey(access, appName, userId, sessionId, filename, version),
              filename,
              mediaType,
              new ByteArrayInputStream(data),
              data.length,
              metadata);
      try {
        Artifact saved = engine.ingest(request);
        Optional<Integer> savedVersion = versionOf(saved);
        if (matchesLogicalName(saved, appName, userId, sessionId, filename)
            && savedVersion.orElse(-1) == version) {
          return version;
        }
        throw new IllegalStateException(
            "Content deduplication resolved this ADK revision to another logical artifact. "
                + "Use an engine policy that preserves logical artifact revisions.");
      } catch (IdempotencyConflictException concurrentSave) {
        if (attempt == MAX_SAVE_ATTEMPTS - 1) {
          throw concurrentSave;
        }
      }
    }
    throw new IllegalStateException("Artifact version allocation did not converge");
  }

  private Part readPart(AccessContext access, Artifact artifact) throws IOException {
    if (artifact.sizeBytes() > maxInlineBytes) {
      throw new IllegalArgumentException("Artifact exceeds the configured inline byte limit");
    }
    try (InputStream content =
        engine.content(access.principalId(), access.tenantId(), artifact.id())) {
      byte[] bytes = content.readNBytes(maxInlineBytes + 1);
      if (bytes.length > maxInlineBytes) {
        throw new IllegalArgumentException("Artifact exceeds the configured inline byte limit");
      }
      return Part.fromBytes(bytes, artifact.mediaType());
    }
  }

  private int nextVersion(
      AccessContext access, String appName, String userId, String sessionId, String filename) {
    return matchingArtifacts(access, appName, userId, sessionId, filename).stream()
        .map(NotifyArtifactService::versionOf)
        .flatMap(Optional::stream)
        .max(Comparator.naturalOrder())
        .map(version -> Math.addExact(version, 1))
        .orElse(0);
  }

  private Optional<Artifact> findArtifact(
      AccessContext access,
      String appName,
      String userId,
      String sessionId,
      String filename,
      @Nullable Integer version) {
    if (version != null && version < 0) {
      return Optional.empty();
    }
    Comparator<Artifact> byVersion =
        Comparator.comparingInt(artifact -> versionOf(artifact).orElse(-1));
    return matchingArtifacts(access, appName, userId, sessionId, filename).stream()
        .filter(artifact -> version == null || versionOf(artifact).orElse(-1) == version)
        .max(byVersion);
  }

  private List<Artifact> matchingArtifacts(
      AccessContext access, String appName, String userId, String sessionId, String filename) {
    return visibleArtifacts(access, appName, userId, sessionId).stream()
        .filter(artifact -> filename.equals(artifact.metadata().get(META_FILENAME)))
        .filter(
            artifact ->
                isUserScoped(filename) == USER_SCOPE.equals(artifact.metadata().get(META_SCOPE)))
        .toList();
  }

  private List<Artifact> visibleArtifacts(
      AccessContext access, String appName, String userId, String sessionId) {
    return engine.listMetadata(access.principalId(), access.tenantId(), catalogLimit).stream()
        .filter(artifact -> isVisible(artifact, appName, userId, sessionId))
        .toList();
  }

  private static boolean isVisible(
      Artifact artifact, String appName, String userId, String sessionId) {
    Map<String, String> metadata = artifact.metadata();
    if (!ARTIFACT_KIND.equals(metadata.get(META_KIND))
        || !appName.equals(metadata.get(META_APP))
        || !userId.equals(metadata.get(META_USER))) {
      return false;
    }
    return USER_SCOPE.equals(metadata.get(META_SCOPE))
        || (SESSION_SCOPE.equals(metadata.get(META_SCOPE))
            && sessionId.equals(metadata.get(META_SESSION)));
  }

  private static boolean matchesLogicalName(
      Artifact artifact, String appName, String userId, String sessionId, String filename) {
    if (!isVisible(artifact, appName, userId, sessionId)) {
      return false;
    }
    Map<String, String> metadata = artifact.metadata();
    return filename.equals(metadata.get(META_FILENAME))
        && (isUserScoped(filename)
            ? USER_SCOPE.equals(metadata.get(META_SCOPE))
            : SESSION_SCOPE.equals(metadata.get(META_SCOPE))
                && sessionId.equals(metadata.get(META_SESSION)));
  }

  private static Optional<Integer> versionOf(Artifact artifact) {
    try {
      String value = artifact.metadata().get(META_VERSION);
      return value == null ? Optional.empty() : Optional.of(Integer.parseInt(value));
    } catch (NumberFormatException invalidVersion) {
      return Optional.empty();
    }
  }

  private static Map<String, String> referenceMetadata(
      String appName, String userId, String sessionId, String filename, int version) {
    Map<String, String> metadata = new LinkedHashMap<>();
    metadata.put(META_KIND, ARTIFACT_KIND);
    metadata.put(META_APP, appName);
    metadata.put(META_USER, userId);
    metadata.put(META_SCOPE, isUserScoped(filename) ? USER_SCOPE : SESSION_SCOPE);
    metadata.put(META_SESSION, isUserScoped(filename) ? "" : sessionId);
    metadata.put(META_FILENAME, filename);
    metadata.put(META_VERSION, Integer.toString(version));
    return Map.copyOf(metadata);
  }

  private static String idempotencyKey(
      AccessContext access,
      String appName,
      String userId,
      String sessionId,
      String filename,
      int version) {
    String scope = isUserScoped(filename) ? USER_SCOPE : sessionId;
    return "adk-"
        + Checksum.sha256(
            String.join(
                "\u0000",
                access.tenantId(),
                appName,
                userId,
                scope,
                filename,
                Integer.toString(version)));
  }

  private AccessContext access(String appName, String userId, String sessionId) {
    requireCoordinate(appName, "appName");
    requireCoordinate(userId, "userId");
    requireCoordinate(sessionId, "sessionId");
    return Objects.requireNonNull(
        accessResolver.resolve(appName, userId, sessionId), "accessResolver result");
  }

  private static String requireFilename(String filename) {
    requireCoordinate(filename, "filename");
    if (filename.length() > 512
        || filename.indexOf('/') >= 0
        || filename.indexOf('\\') >= 0
        || filename.chars().anyMatch(character -> Character.isISOControl(character))) {
      throw new IllegalArgumentException("filename contains unsafe characters");
    }
    return filename;
  }

  private static void requireCoordinate(String value, String name) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(name + " is required");
    }
  }

  private static boolean isUserScoped(String filename) {
    return filename.startsWith(USER_NAMESPACE_PREFIX);
  }

  /** Resolves trusted engine authorization coordinates for an ADK artifact operation. */
  @FunctionalInterface
  public interface AccessResolver {
    AccessContext resolve(String appName, String userId, String sessionId);
  }

  /** Tenant and principal sent to the artifact engine authorization boundary. */
  public record AccessContext(String tenantId, String principalId) {
    public AccessContext {
      requireCoordinate(tenantId, "tenantId");
      requireCoordinate(principalId, "principalId");
    }
  }
}
