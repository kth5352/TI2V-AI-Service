// frontend/src/App.jsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import "./styles.css";

const API = "/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null); // 드롭존 썸네일 표시용
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(4);
  const [fps, setFps] = useState(16);
  const [width, setWidth] = useState(1280);
  const [height, setHeight] = useState(704);
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const [videoUrl, setVideoUrl] = useState(null);
  const [negPrompt, setNegPrompt] = useState("");

  // 에러 표시
  const [errMsg, setErrMsg] = useState(null);
  const [errCode, setErrCode] = useState(null);

  // --- 폴링 제어 (지수 백오프 & 탭 비활성화 시 일시정지) ---
  const baseDelay = 1500; // 1.5s 시작
  const maxDelay = 6000; // 최대 6s
  const delayRef = useRef(baseDelay);
  const timerRef = useRef(null);
  const abortRef = useRef(null);

  // --- SSE 사용 여부 (성공 시 true, 실패 시 false로 폴백 폴링) ---
  const [useSSEOk, setUseSSEOk] = useState(false);
  const sseRef = useRef(null);

  // --- 이미지 드롭/선택 ---
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles?.[0]) {
      const f = acceptedFiles[0];
      setFile(f);

      // 미리보기 URL 생성
      const url = URL.createObjectURL(f);
      setPreviewUrl(url);

      // 이미지 실제 해상도 읽어 width/height 자동 설정 (32배수 스냅)
      const img = new Image();
      img.onload = () => {
        const snappedW = Math.max(320, Math.floor(img.naturalWidth / 32) * 32);
        const snappedH = Math.max(320, Math.floor(img.naturalHeight / 32) * 32);
        setWidth(snappedW);
        setHeight(snappedH);
      };
      img.src = url;
    }
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    multiple: false,
    onDrop,
  });

  // 미리보기 URL 메모리 정리
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const canSubmit = useMemo(
    () => prompt.trim().length > 0 || !!file,
    [prompt, file]
  );

  async function handleGenerate() {
    setStatus("submitting");
    setProgress(0);
    setVideoUrl(null);
    setUseSSEOk(false); // 새 작업 시작 시 초기화
    delayRef.current = baseDelay;
    setErrMsg(null);
    setErrCode(null);

    const fd = new FormData();
    fd.append("prompt", prompt);
    fd.append("duration", String(duration));
    fd.append("fps", String(fps));
    fd.append("width", String(width));
    fd.append("height", String(height));
    if (negPrompt) fd.append("negative_prompt", negPrompt);
    if (file) fd.append("image", file);

    const r = await fetch(`${API}/generate`, { method: "POST", body: fd });
    if (!r.ok) {
      setStatus("error");
      setErrCode(`HTTP_${r.status}`);
      setErrMsg("작업 생성에 실패했습니다.");
      return;
    }
    const { job_id } = await r.json();
    setJobId(job_id);
    setStatus("running");
  }

  // --- SSE 구독 (가능하면 서버 푸시 방식으로 진행률 수신) ---
  useEffect(() => {
    if (!jobId) return;

    if (!window.EventSource) {
      setUseSSEOk(false);
      return;
    }

    try {
      const es = new EventSource(`${API}/stream/${jobId}`);
      sseRef.current = es;

      const onUpdate = (e) => {
        try {
          const js = JSON.parse(e.data);
          const pct =
            typeof js.progress === "number"
              ? js.progress <= 1
                ? Math.round(js.progress * 100)
                : Math.round(js.progress)
              : 0;
          setProgress(Math.min(100, Math.max(0, pct)));
          setStatus(js.status);
          if (js.status === "done" && js.result) setVideoUrl(js.result);
          if (js.status === "error") {
            setErrMsg(js.error || "알 수 없는 오류가 발생했습니다.");
            setErrCode(js.error_code || null);
          }
          setUseSSEOk(true);
        } catch {
          setUseSSEOk(false);
        }
      };

      const onError = () => {
        setUseSSEOk(false);
        if (sseRef.current) {
          sseRef.current.close();
          sseRef.current = null;
        }
      };

      es.addEventListener("update", onUpdate);
      es.addEventListener("error", onError);

      return () => {
        es.removeEventListener("update", onUpdate);
        es.removeEventListener("error", onError);
        es.close();
        sseRef.current = null;
      };
    } catch {
      setUseSSEOk(false);
    }
  }, [jobId]);

  // --- 폴백 폴링 (SSE 실패 or 미지원 시만 실행) ---
  useEffect(() => {
    if (!jobId || useSSEOk) return;

    const poll = async () => {
      if (document.hidden) {
        timerRef.current = setTimeout(
          poll,
          Math.min(maxDelay, delayRef.current * 1.5)
        );
        return;
      }

      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      try {
        const r = await fetch(`${API}/status/${jobId}`, {
          signal: abortRef.current.signal,
        });
        if (!r.ok) throw new Error("status not ok");
        const js = await r.json();
        const pct =
          typeof js.progress === "number"
            ? js.progress <= 1
              ? Math.round(js.progress * 100)
              : Math.round(js.progress)
            : 0;

        setProgress(Math.min(100, Math.max(0, pct)));
        setStatus(js.status);

        if (js.status === "done" && js.result) {
          setVideoUrl(js.result);
          return;
        }
        if (js.status === "error") {
          setErrMsg(js.error || "알 수 없는 오류가 발생했습니다.");
          setErrCode(js.error_code || null);
          return;
        }

        if (pct >= 96 && pct < 100) {
          delayRef.current = baseDelay;
        } else if (pct === 0) {
          delayRef.current = Math.min(maxDelay, delayRef.current * 1.6);
        } else {
          delayRef.current = Math.min(
            maxDelay,
            Math.max(baseDelay, delayRef.current * 1.2)
          );
        }
      } catch {
        delayRef.current = Math.min(maxDelay, delayRef.current * 1.6);
      }

      timerRef.current = setTimeout(poll, delayRef.current);
    };

    timerRef.current = setTimeout(poll, delayRef.current);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, [jobId, useSSEOk]);

  return (
    <div className="container">
      <h2>WAN 2.2 Image → Video</h2>
      <p className="help">
        이미지 없이 프롬프트만 보내면 Text→Video, 이미지를 넣으면 Image→Video로
        동작합니다.
      </p>

      <div className="grid">
        <div className="card">
          <label>이미지 업로드 (드래그&드롭 또는 버튼)</label>
          <div className="dropzone" {...getRootProps()}>
            <input {...getInputProps()} />
            {isDragActive ? (
              <p>여기에 놓으세요…</p>
            ) : previewUrl ? (
              <div className="preview-wrap">
                <img className="preview" src={previewUrl} alt="preview" />
                <div className="preview-meta">
                  선택됨: {file?.name || "이미지"}
                  <br />
                  입력 해상도: {width} × {height}
                </div>
              </div>
            ) : (
              <p>
                이미지를 여기로 드래그하거나 클릭하여 선택
                <br />
                (선택 시 자동으로 이미지 해상도로 입력 크기를 맞춥니다)
              </p>
            )}
          </div>
          <div className="help">
            PNG/JPG/WEBP 권장. 미첨부 시 Text→Video로 생성됩니다.
          </div>

          <div style={{ height: 12 }} />

          <label>프롬프트</label>
          <textarea
            rows={5}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="예) a white cat surfing on a sunny beach, cinematic lighting"
          />

          <label>네거티브 프롬프트(선택)</label>
          <input
            value={negPrompt}
            onChange={(e) => setNegPrompt(e.target.value)}
            placeholder="예) low quality, blurry"
          />
        </div>

        <div className="card">
          <label>길이(초)</label>
          <input
            type="number"
            min={1}
            max={10}
            step={1}
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
          />

          <label>FPS</label>
          <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
            <option value={12}>12</option>
            <option value={16}>16</option>
            <option value={24}>24</option>
          </select>

          <div className="row">
            <div style={{ flex: 1 }}>
              <label>가로(width)</label>
              <input
                type="number"
                min={320}
                max={1920}
                step={16}
                value={width}
                onChange={(e) => setWidth(Number(e.target.value))}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label>세로(height)</label>
              <input
                type="number"
                min={320}
                max={1920}
                step={16}
                value={height}
                onChange={(e) => setHeight(Number(e.target.value))}
              />
            </div>
          </div>
          <div className="help">
            권장 720p TI2V: 1280×704 또는 704×1280. (모델 가이드)
          </div>

          <div style={{ height: 18 }} />
          <button
            disabled={!canSubmit || status === "running"}
            onClick={handleGenerate}>
            {status === "running" ? "생성 중..." : "동영상 생성"}
          </button>

          <div style={{ height: 18 }} />
          <div
            className="progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
            title={useSSEOk ? "SSE 스트리밍" : "폴링"}>
            <div style={{ width: `${progress}%` }} />
          </div>
          <div className="help">
            진행률: {progress}%{" "}
            {useSSEOk
              ? "(SSE)"
              : `(폴링 ~${Math.round(delayRef.current / 100) / 10}s)`}
          </div>
        </div>
      </div>

      <div style={{ height: 24 }} />

      {status === "error" && (
        <div className="card error">
          <strong>에러</strong>
          <div className="err-code">
            {errCode ? <code>{errCode}</code> : null}
          </div>
          <div className="err-msg">
            {errMsg || "에러가 발생했습니다. 백엔드 로그를 확인해주세요."}
          </div>
        </div>
      )}

      {videoUrl && (
        <div className="card">
          <h3>결과</h3>
          <video className="video" src={videoUrl} controls />
          <div className="spacer-12" />
          <a href={videoUrl} download>
            <button>다운로드</button>
          </a>
          <div className="help">
            서버의 <code>backend/downloads</code> 폴더에도 저장됩니다.
          </div>
        </div>
      )}
    </div>
  );
}
