// Cliente da API do conversor. Usa XMLHttpRequest (em vez de fetch) para
// expor progresso de upload/download, útil com arquivos HDF5 grandes.

export interface Hdf5Node {
  path: string;
  type: "group" | "dataset";
  shape?: number[];
  dtype?: string;
  size?: number;
  byte_size?: number;
  attrs?: number;
  is_numeric?: boolean;
}

export interface InspectResponse {
  filename: string;
  size_bytes: number;
  root_attrs: Record<string, string>;
  nodes: Hdf5Node[];
}

export interface Progress {
  phase: "upload" | "processing" | "download";
  /** 0–100, ou null quando o total é desconhecido */
  percent: number | null;
}

export interface ConvertResult {
  blob: Blob;
  filename: string;
}

interface RequestOptions {
  responseType?: "json" | "blob";
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
}

// URL base da API: vazio usa o proxy do Vite para http://127.0.0.1:8000
const rawBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");
const API_BASE_URL = rawBaseUrl ? `${rawBaseUrl}/api` : "/api";

export class AbortedError extends Error {
  constructor() {
    super("Operação cancelada.");
    this.name = "AbortedError";
  }
}

function parseFilenameFromHeader(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const match = header.match(/filename\*?=['"]?(?:UTF-\d['"]*)?([^;\r\n"']*)['"]?/i);
  return match?.[1] ? decodeURIComponent(match[1]) : fallback;
}

async function extractErrorMessage(xhr: XMLHttpRequest): Promise<string> {
  if (xhr.status === 408) {
    return "Tempo limite esgotado (Erro 408). O servidor demorou para responder ao envio do arquivo. Verifique se o backend local está ativo em http://127.0.0.1:8000 (ou aguarde se a API remota estiver iniciando).";
  }
  const fallback = `Erro ${xhr.status} no servidor.`;
  try {
    const raw: unknown = xhr.response;
    const text =
      raw instanceof Blob ? await raw.text() : typeof raw === "string" ? raw : JSON.stringify(raw);
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (Array.isArray(parsed.detail)) return parsed.detail.map((d) => d?.msg ?? String(d)).join("; ");
    return fallback;
  } catch {
    return fallback;
  }
}

interface PostResult<T> {
  data: T;
  filename?: string;
}

function postForm<T>(path: string, form: FormData, opts: RequestOptions = {}): Promise<PostResult<T>> {
  return new Promise<PostResult<T>>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}${path}`);
    xhr.responseType = opts.responseType ?? "json";

    xhr.upload.onprogress = (e) => {
      opts.onProgress?.({
        phase: "upload",
        percent: e.lengthComputable ? Math.round((e.loaded / e.total) * 100) : null,
      });
    };
    // Upload terminou: o servidor está processando o HDF5
    xhr.upload.onload = () => opts.onProgress?.({ phase: "processing", percent: null });
    xhr.onprogress = (e) => {
      if (xhr.readyState < XMLHttpRequest.LOADING) return;
      opts.onProgress?.({
        phase: "download",
        percent: e.lengthComputable ? Math.round((e.loaded / e.total) * 100) : null,
      });
    };

    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const disposition = xhr.getResponseHeader("content-disposition");
        const filename = disposition ? parseFilenameFromHeader(disposition, "") : undefined;
        resolve({ data: xhr.response as T, filename });
      } else {
        reject(new Error(await extractErrorMessage(xhr)));
      }
    };
    xhr.onerror = () =>
      reject(
        new Error(
          "Falha de conexão com o servidor Python. Verifique se o backend local está rodando em http://127.0.0.1:8000.",
        ),
      );
    xhr.onabort = () => reject(new AbortedError());

    if (opts.signal) {
      if (opts.signal.aborted) {
        reject(new AbortedError());
        return;
      }
      opts.signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }

    xhr.send(form);
  });
}

export async function inspectHdf5(file: File, opts: Omit<RequestOptions, "responseType"> = {}) {
  const form = new FormData();
  form.append("file", file);
  const res = await postForm<InspectResponse>("/inspect", form, { ...opts, responseType: "json" });
  return res.data;
}

export async function convertHdf5ToRaw(
  file: File,
  targetDataset: string,
  opts: Omit<RequestOptions, "responseType"> = {},
): Promise<ConvertResult> {
  const form = new FormData();
  form.append("file", file);
  if (targetDataset) form.append("target_dataset", targetDataset);
  const res = await postForm<Blob>("/convert", form, { ...opts, responseType: "blob" });
  const defaultExt = targetDataset ? ".raw" : ".zip";
  const baseName = file.name.replace(/\.(hdf5|h5)$/i, "");
  const fallbackFilename = `${baseName}${defaultExt}`;
  return {
    blob: res.data,
    filename: res.filename || fallbackFilename,
  };
}

// Alias para manter compatibilidade
export const convertHdf5ToHal = convertHdf5ToRaw;

/** Dispara o download de um Blob no navegador. */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Pequeno atraso garante que o navegador iniciou o download antes de liberar a URL
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
