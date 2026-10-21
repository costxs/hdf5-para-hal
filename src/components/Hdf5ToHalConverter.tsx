import React, { useMemo, useRef, useState } from "react";
import {
  AbortedError,
  convertHdf5ToHal,
  downloadBlob,
  inspectHdf5,
  type Hdf5Node,
  type InspectResponse,
  type Progress,
} from "../api/hdf5";

type ConversionStatus = "idle" | "inspecting" | "ready" | "converting" | "success" | "error";

const VALID_EXTENSIONS = [".h5", ".hdf5"];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ${units[i]}`;
}

function progressLabel(p: Progress | null, action: "inspect" | "convert"): string {
  if (!p) return action === "inspect" ? "Preparando leitura..." : "Preparando conversão...";
  if (p.phase === "upload") return `Enviando arquivo${p.percent !== null ? ` — ${p.percent}%` : "..."}`;
  if (p.phase === "processing")
    return action === "inspect" ? "Lendo estrutura HDF5..." : "Convertendo para .HAL...";
  return `Baixando .hal${p.percent !== null ? ` — ${p.percent}%` : "..."}`;
}

// --------------------------------------------------------------------------- //
// Subcomponentes
// --------------------------------------------------------------------------- //
const ProgressBar: React.FC<{ progress: Progress | null; action: "inspect" | "convert" }> = ({
  progress,
  action,
}) => {
  const indeterminate = !progress || progress.percent === null || progress.phase === "processing";
  const width = indeterminate ? 100 : progress!.percent!;
  return (
    <div className="space-y-2 animate-fade-up" role="status" aria-live="polite">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-300">{progressLabel(progress, action)}</span>
        {!indeterminate && <span className="font-mono text-indigo-300">{width}%</span>}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className={`relative h-full rounded-full bg-gradient-to-r from-indigo-500 via-violet-500 to-cyan-400 transition-[width] duration-300 ${
            indeterminate ? "opacity-70" : ""
          }`}
          style={{ width: `${width}%` }}
        >
          <div className="progress-shimmer absolute inset-0 rounded-full" />
        </div>
      </div>
    </div>
  );
};

const TreeRow: React.FC<{
  node: Hdf5Node;
  selected: boolean;
  onSelect: (path: string) => void;
  disabled: boolean;
}> = ({ node, selected, onSelect, disabled }) => {
  const depth = node.path.split("/").length - 2;
  const name = node.path.split("/").pop() || "/";
  const isDataset = node.type === "dataset";

  const content = (
    <>
      <span className={`shrink-0 ${isDataset ? "text-cyan-400" : "text-amber-400"}`} aria-hidden>
        {isDataset ? (
          <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
            <path d="M10 2c-3.9 0-7 1.3-7 3v10c0 1.7 3.1 3 7 3s7-1.3 7-3V5c0-1.7-3.1-3-7-3Zm0 2c3.3 0 5 .9 5 1s-1.7 1-5 1-5-.9-5-1 1.7-1 5-1Z" />
          </svg>
        ) : (
          <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
            <path d="M2 5a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5Z" />
          </svg>
        )}
      </span>
      <span className={`truncate font-mono text-sm ${isDataset ? "text-slate-200" : "text-slate-400"}`}>
        {name}
      </span>
      {isDataset && (
        <span className="ml-auto flex shrink-0 items-center gap-2 font-mono text-[11px] text-slate-500">
          <span>[{node.shape?.join(" × ") || "escalar"}]</span>
          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-400">{node.dtype}</span>
        </span>
      )}
    </>
  );

  const padding = { paddingLeft: `${depth * 18 + 12}px` };

  if (!isDataset) {
    return (
      <li className="flex items-center gap-2 py-1.5 pr-3" style={padding}>
        {content}
      </li>
    );
  }

  return (
    <li>
      <button
        type="button"
        id={`dataset-${node.path.replace(/[^a-zA-Z0-9]/g, "_")}`}
        disabled={disabled}
        onClick={() => onSelect(selected ? "" : node.path)}
        aria-pressed={selected}
        className={`flex w-full items-center gap-2 rounded-lg py-1.5 pr-3 text-left transition-colors disabled:cursor-not-allowed ${
          selected
            ? "bg-indigo-500/15 ring-1 ring-indigo-500/50"
            : "hover:bg-slate-800/70"
        }`}
        style={padding}
      >
        {content}
      </button>
    </li>
  );
};

// --------------------------------------------------------------------------- //
// Componente principal
// --------------------------------------------------------------------------- //
export const Hdf5ToHalConverter: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [info, setInfo] = useState<InspectResponse | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [status, setStatus] = useState<ConversionStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [filter, setFilter] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const busy = status === "inspecting" || status === "converting";
  const nodes = useMemo(() => info?.nodes ?? [], [info]);
  const datasetsCount = useMemo(() => nodes.filter((n) => n.type === "dataset").length, [nodes]);
  const visibleNodes = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return nodes;
    return nodes.filter((n) => n.path.toLowerCase().includes(q));
  }, [nodes, filter]);

  const handleError = (err: unknown, fallback: string) => {
    if (err instanceof AbortedError) {
      setStatus(info ? "ready" : "idle");
      setProgress(null);
      return;
    }
    setErrorMessage(err instanceof Error ? err.message : fallback);
    setStatus("error");
    setProgress(null);
  };

  const validateAndLoadFile = async (selectedFile: File) => {
    const isValid = VALID_EXTENSIONS.some((ext) => selectedFile.name.toLowerCase().endsWith(ext));
    if (!isValid) {
      setErrorMessage("Arquivo inválido. Selecione um arquivo .h5 ou .hdf5.");
      setStatus("error");
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setFile(selectedFile);
    setInfo(null);
    setSelectedDataset("");
    setFilter("");
    setErrorMessage(null);
    setProgress(null);
    setStatus("inspecting");

    try {
      const data = await inspectHdf5(selectedFile, { onProgress: setProgress, signal: controller.signal });
      setInfo(data);
      setStatus("ready");
      setProgress(null);
    } catch (err) {
      if (err instanceof AbortedError) {
        setStatus("idle");
        setProgress(null);
        return;
      }
      handleError(err, "Falha ao inspecionar o arquivo HDF5.");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) void validateAndLoadFile(f);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (busy) return;
    const f = e.dataTransfer.files?.[0];
    if (f) void validateAndLoadFile(f);
  };

  const handleConvert = async () => {
    if (!file) return;
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("converting");
    setErrorMessage(null);
    setProgress(null);

    try {
      const blob = await convertHdf5ToHal(file, selectedDataset, {
        onProgress: setProgress,
        signal: controller.signal,
      });
      const baseName = file.name.replace(/\.(hdf5|h5)$/i, "");
      downloadBlob(blob, `${baseName}.hal`);
      setStatus("success");
      setProgress(null);
    } catch (err) {
      handleError(err, "Erro inesperado na conversão.");
    }
  };

  const handleCancel = () => abortRef.current?.abort();

  const resetState = () => {
    abortRef.current?.abort();
    setFile(null);
    setInfo(null);
    setSelectedDataset("");
    setFilter("");
    setStatus("idle");
    setErrorMessage(null);
    setProgress(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 font-sans text-slate-100">
      {/* Fundo decorativo */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div className="animate-glow absolute -top-40 left-1/2 h-[480px] w-[480px] -translate-x-1/2 rounded-full bg-indigo-600/25 blur-3xl" />
        <div className="animate-glow absolute -bottom-40 -right-20 h-[380px] w-[380px] rounded-full bg-cyan-500/15 blur-3xl [animation-delay:-3s]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgb(148_163_184/0.05)_1px,transparent_1px),linear-gradient(90deg,rgb(148_163_184/0.05)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_75%)]" />
      </div>

      <main className="relative flex min-h-screen items-center justify-center p-6">
        <section className="w-full max-w-3xl space-y-6 rounded-2xl border border-white/10 bg-slate-900/70 p-8 shadow-2xl shadow-indigo-950/40 backdrop-blur-xl">
          {/* Cabeçalho */}
          <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/10 pb-5">
            <div>
              <h1 className="bg-gradient-to-r from-white via-indigo-100 to-cyan-200 bg-clip-text text-2xl font-bold tracking-tight text-transparent">
                Conversor HDF5 para .HAL
              </h1>
              <p className="mt-1 text-sm text-slate-400">
                Inspecione datasets hierárquicos e exporte para a estrutura .hal
              </p>
            </div>
            <span className="rounded-full border border-indigo-500/30 bg-indigo-500/10 px-3 py-1 font-mono text-xs text-indigo-300">
              .h5 / .hdf5 → .hal
            </span>
          </header>

          {/* Zona de Drag and Drop */}
          <div
            id="dropzone"
            role="button"
            tabIndex={0}
            aria-label="Selecionar arquivo HDF5"
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => !busy && fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && !busy) {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`group rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              busy ? "cursor-progress" : "cursor-pointer"
            } ${
              isDragging
                ? "scale-[1.01] border-indigo-400 bg-indigo-500/10 shadow-lg shadow-indigo-500/20"
                : "border-slate-700 bg-slate-950/40 hover:border-indigo-500/60 hover:bg-slate-950/70"
            }`}
          >
            <input
              id="file-input"
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".h5,.hdf5"
              className="hidden"
            />

            {!file ? (
              <div className="space-y-3">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-500/15 text-indigo-300 transition-transform duration-200 group-hover:-translate-y-0.5">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="h-6 w-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                  </svg>
                </div>
                <div className="font-medium text-slate-300">
                  Arraste seu arquivo <span className="text-indigo-300">.hdf5</span> aqui ou clique para selecionar
                </div>
                <p className="text-xs text-slate-500">Suporta arquivos científicos e industriais estruturados em HDF5</p>
              </div>
            ) : (
              <div className="flex items-center justify-between gap-4 rounded-lg border border-white/10 bg-slate-900/80 px-4 py-3">
                <div className="min-w-0 text-left">
                  <p className="truncate text-sm font-semibold text-slate-200">{file.name}</p>
                  <p className="text-xs text-slate-500">
                    {formatBytes(file.size)}
                    {info && ` · ${datasetsCount} datasets · ${nodes.length - datasetsCount} grupos`}
                  </p>
                </div>
                <button
                  id="remove-file"
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    resetState();
                  }}
                  className="shrink-0 rounded-md bg-rose-500/10 px-3 py-1 text-xs text-rose-300 transition-colors hover:bg-rose-500/20"
                >
                  Remover
                </button>
              </div>
            )}
          </div>

          {/* Progresso */}
          {busy && <ProgressBar progress={progress} action={status === "inspecting" ? "inspect" : "convert"} />}

          {/* Árvore do HDF5 */}
          {info && nodes.length > 0 && (
            <div className="animate-fade-up space-y-3 rounded-xl border border-white/10 bg-slate-950/50 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Estrutura do arquivo ({datasetsCount} datasets)
                </h2>
                <input
                  id="tree-filter"
                  type="search"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="Filtrar caminho..."
                  className="w-48 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
                />
              </div>

              <button
                id="dataset-all"
                type="button"
                disabled={busy}
                onClick={() => setSelectedDataset("")}
                aria-pressed={selectedDataset === ""}
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                  selectedDataset === ""
                    ? "bg-indigo-500/15 text-indigo-200 ring-1 ring-indigo-500/50"
                    : "text-slate-300 hover:bg-slate-800/70"
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${selectedDataset === "" ? "bg-indigo-400" : "bg-slate-600"}`} />
                Converter todos os datasets do arquivo
              </button>

              <ul className="max-h-72 space-y-0.5 overflow-y-auto pr-1 [scrollbar-color:#334155_transparent] [scrollbar-width:thin]">
                {visibleNodes.map((node) => (
                  <TreeRow
                    key={node.path}
                    node={node}
                    selected={selectedDataset === node.path}
                    onSelect={setSelectedDataset}
                    disabled={busy}
                  />
                ))}
                {visibleNodes.length === 0 && (
                  <li className="px-3 py-4 text-center text-xs text-slate-500">Nenhum caminho corresponde ao filtro.</li>
                )}
              </ul>

              {Object.keys(info.root_attrs).length > 0 && (
                <details className="text-xs text-slate-400">
                  <summary className="cursor-pointer select-none hover:text-slate-200">
                    Atributos da raiz ({Object.keys(info.root_attrs).length})
                  </summary>
                  <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono">
                    {Object.entries(info.root_attrs).map(([k, v]) => (
                      <React.Fragment key={k}>
                        <dt className="text-slate-500">{k}</dt>
                        <dd className="truncate text-slate-300">{v}</dd>
                      </React.Fragment>
                    ))}
                  </dl>
                </details>
              )}
            </div>
          )}

          {info && nodes.length === 0 && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
              O arquivo não contém grupos nem datasets.
            </div>
          )}

          {/* Feedback de Erro */}
          {errorMessage && (
            <div role="alert" className="animate-fade-up rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
              {errorMessage}
            </div>
          )}

          {/* Feedback de Sucesso */}
          {status === "success" && (
            <div className="animate-fade-up rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-300">
              Conversão concluída! O download do arquivo <strong>.hal</strong> foi iniciado automaticamente.
            </div>
          )}

          {/* Ações */}
          <div className="flex gap-3 pt-1">
            {busy && (
              <button
                id="cancel-button"
                type="button"
                onClick={handleCancel}
                className="rounded-xl border border-slate-700 px-5 py-3 text-sm font-semibold text-slate-300 transition-colors hover:border-rose-500/50 hover:text-rose-300"
              >
                Cancelar
              </button>
            )}
            <button
              id="convert-button"
              type="button"
              disabled={!file || !info || busy || datasetsCount === 0}
              onClick={handleConvert}
              className="flex-1 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-600/25 transition-all hover:-translate-y-0.5 hover:shadow-indigo-500/40 active:translate-y-0 disabled:translate-y-0 disabled:cursor-not-allowed disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 disabled:shadow-none"
            >
              {status === "inspecting" && "Lendo estrutura HDF5..."}
              {status === "converting" && "Convertendo para .HAL..."}
              {!busy && (selectedDataset ? `Converter ${selectedDataset} e baixar .HAL` : "Converter e Baixar .HAL")}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Hdf5ToHalConverter;
