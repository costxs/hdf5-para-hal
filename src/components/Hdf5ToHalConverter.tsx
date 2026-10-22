import React, { useMemo, useRef, useState } from "react";
import {
  AbortedError,
  convertHdf5ToRaw,
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
    return action === "inspect" ? "Lendo estrutura HDF5..." : "Extraindo buffer binário .RAW...";
  return `Baixando arquivo${p.percent !== null ? ` — ${p.percent}%` : "..."}`;
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
        {!indeterminate && <span className="font-mono text-cyan-300">{width}%</span>}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className={`relative h-full rounded-full bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-400 transition-[width] duration-300 ${
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
          {node.byte_size !== undefined && (
            <span className="rounded bg-cyan-950/60 px-1.5 py-0.5 text-cyan-300">
              {formatBytes(node.byte_size)}
            </span>
          )}
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
            ? "bg-cyan-500/15 ring-1 ring-cyan-500/50"
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
  const [copiedInfo, setCopiedInfo] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const busy = status === "inspecting" || status === "converting";
  const nodes = useMemo(() => info?.nodes ?? [], [info]);
  const datasets = useMemo(() => nodes.filter((n) => n.type === "dataset"), [nodes]);
  const datasetsCount = datasets.length;

  const selectedNode = useMemo(
    () => datasets.find((d) => d.path === selectedDataset),
    [datasets, selectedDataset],
  );

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
      // Se houver exatamente 1 dataset no arquivo, já o deixa pré-selecionado
      const dsets = data.nodes.filter((n) => n.type === "dataset");
      if (dsets.length === 1) {
        setSelectedDataset(dsets[0].path);
      }
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
      const { blob, filename } = await convertHdf5ToRaw(file, selectedDataset, {
        onProgress: setProgress,
        signal: controller.signal,
      });
      downloadBlob(blob, filename);
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

  const copyMetadata = () => {
    if (!selectedNode) return;
    const meta = [
      `Dataset: ${selectedNode.path}`,
      `Shape: ${selectedNode.shape?.join(" x ") || "scalar"}`,
      `Dtype: ${selectedNode.dtype}`,
      `Byte Size: ${selectedNode.byte_size ?? 0} bytes`,
      `Format: Raw Binary (C-contiguous / Row-major)`,
    ].join("\n");
    void navigator.clipboard.writeText(meta);
    setCopiedInfo(true);
    setTimeout(() => setCopiedInfo(false), 2000);
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 font-sans text-slate-100">
      {/* Fundo decorativo */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div className="animate-glow absolute -top-40 left-1/2 h-[480px] w-[480px] -translate-x-1/2 rounded-full bg-cyan-600/20 blur-3xl" />
        <div className="animate-glow absolute -bottom-40 -right-20 h-[380px] w-[380px] rounded-full bg-indigo-500/15 blur-3xl [animation-delay:-3s]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgb(148_163_184/0.05)_1px,transparent_1px),linear-gradient(90deg,rgb(148_163_184/0.05)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_75%)]" />
      </div>

      <main className="relative flex min-h-screen items-center justify-center p-6">
        <section className="w-full max-w-3xl space-y-6 rounded-2xl border border-white/10 bg-slate-900/70 p-8 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl">
          {/* Cabeçalho */}
          <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/10 pb-5">
            <div>
              <h1 className="bg-gradient-to-r from-white via-cyan-100 to-indigo-200 bg-clip-text text-2xl font-bold tracking-tight text-transparent">
                Conversor HDF5 para Binário .RAW
              </h1>
              <p className="mt-1 text-sm text-slate-400">
                Exporte datasets do HDF5 diretamente como buffer binário puro (<code className="text-cyan-300">.raw</code>) para visualização 3D, petrofísica e sísmica
              </p>
            </div>
            <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 font-mono text-xs text-cyan-300">
              .h5 / .hdf5 → .raw
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
            className={`group rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 ${
              busy ? "cursor-progress" : "cursor-pointer"
            } ${
              isDragging
                ? "scale-[1.01] border-cyan-400 bg-cyan-500/10 shadow-lg shadow-cyan-500/20"
                : "border-slate-700 bg-slate-950/40 hover:border-cyan-500/60 hover:bg-slate-950/70"
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
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300 transition-transform duration-200 group-hover:-translate-y-0.5">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="h-6 w-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                  </svg>
                </div>
                <div className="font-medium text-slate-300">
                  Arraste seu arquivo <span className="text-cyan-300">.h5 / .hdf5</span> aqui ou clique para selecionar
                </div>
                <p className="text-xs text-slate-500">
                  Processamento via streaming com suporte a arquivos volumosos
                </p>
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
            <div className="animate-fade-up space-y-4 rounded-xl border border-white/10 bg-slate-950/50 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Datasets Disponíveis ({datasetsCount})
                </h2>
                <input
                  id="tree-filter"
                  type="search"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="Filtrar por nome/caminho..."
                  className="w-52 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              {/* Botão de seleção de todos (ZIP) */}
              <button
                id="dataset-all"
                type="button"
                disabled={busy}
                onClick={() => setSelectedDataset("")}
                aria-pressed={selectedDataset === ""}
                className={`flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                  selectedDataset === ""
                    ? "bg-cyan-500/15 text-cyan-200 ring-1 ring-cyan-500/50"
                    : "text-slate-300 hover:bg-slate-800/70"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${selectedDataset === "" ? "bg-cyan-400" : "bg-slate-600"}`} />
                  <span>Todos os datasets numéricos</span>
                </div>
                <span className="font-mono text-xs text-slate-500">(.zip com arquivos .raw)</span>
              </button>

              <ul className="max-h-64 space-y-0.5 overflow-y-auto pr-1 [scrollbar-color:#334155_transparent] [scrollbar-width:thin]">
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

              {/* Card de Informações Técnicas para Software 3D */}
              {selectedNode && (
                <div className="rounded-lg border border-cyan-500/30 bg-cyan-950/20 p-3.5 text-xs text-slate-300 animate-fade-up">
                  <div className="flex items-center justify-between pb-2 border-b border-cyan-500/20">
                    <span className="font-semibold text-cyan-200">
                      Parâmetros para softwares 3D (ImageJ, ParaView, Petrel, etc.):
                    </span>
                    <button
                      type="button"
                      onClick={copyMetadata}
                      className="rounded bg-cyan-500/20 px-2 py-0.5 font-mono text-[11px] text-cyan-300 hover:bg-cyan-500/30 transition-colors"
                    >
                      {copiedInfo ? "Copiado!" : "Copiar metadados"}
                    </button>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2.5 font-mono text-[11px]">
                    <div>
                      <span className="text-slate-500 block">Shape / Dimensões</span>
                      <span className="text-cyan-300 font-semibold">{selectedNode.shape?.join(" × ") || "Escalar"}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Data Type</span>
                      <span className="text-slate-200 font-semibold">{selectedNode.dtype}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Tamanho Binário</span>
                      <span className="text-slate-200">{formatBytes(selectedNode.byte_size ?? 0)}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Ordem dos Bytes</span>
                      <span className="text-slate-300">C-order (Little Endian)</span>
                    </div>
                  </div>
                </div>
              )}

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
              Conversão concluída! O arquivo binário puro <strong>.raw</strong> foi gerado e o download iniciado.
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
              className="flex-1 rounded-xl bg-gradient-to-r from-cyan-600 via-indigo-600 to-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-cyan-600/25 transition-all hover:-translate-y-0.5 hover:shadow-cyan-500/40 active:translate-y-0 disabled:translate-y-0 disabled:cursor-not-allowed disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 disabled:shadow-none"
            >
              {status === "inspecting" && "Lendo estrutura HDF5..."}
              {status === "converting" && "Gerando arquivo binário .RAW..."}
              {!busy &&
                (selectedDataset
                  ? `Exportar dataset ${selectedDataset} (.RAW)`
                  : `Exportar todos os datasets (.ZIP com .RAW)`)}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Hdf5ToHalConverter;
