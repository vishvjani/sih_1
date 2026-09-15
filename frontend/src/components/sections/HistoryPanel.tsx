import React, { useEffect, useState, useCallback } from 'react';
import { getAnalysisHistory, getAnalysisById, type HistorySummary } from '../../services/api';
import type { AnalysisResult } from '../../types';
import {
  History,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  ChevronRight,
  Clock,
  Database,
  Loader2,
  ServerCrash,
  Inbox,
} from 'lucide-react';

interface HistoryPanelProps {
  onLoadResult: (result: AnalysisResult) => void;
  backendOnline: boolean;
}

function formatDate(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function ConfidenceBadge({ value, isAi }: { value: number; isAi: boolean }) {
  const pct = Math.round(value * 100);
  const color = isAi
    ? 'text-red-400 border-red-500/30 bg-red-500/10'
    : 'text-[#96E071] border-[#96E071]/30 bg-[#96E071]/10';
  return (
    <span className={`font-mono text-xs px-2 py-0.5 rounded border ${color}`}>
      {pct}%
    </span>
  );
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ onLoadResult, backendOnline }) => {
  const [records, setRecords] = useState<HistorySummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAnalysisHistory(30);
      setRecords(data.results);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load history.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (backendOnline) fetchHistory();
  }, [backendOnline, fetchHistory]);

  const handleLoad = async (id: string) => {
    setLoadingId(id);
    try {
      const full = await getAnalysisById(id);
      // Map DB record to AnalysisResult shape
      const result: AnalysisResult = {
        id: full.id,
        image_name: full.image_name,
        prediction: full.prediction,
        is_ai_generated: full.is_ai_generated,
        raw_probability_ai: full.raw_probability_ai,
        calibrated_confidence: full.calibrated_confidence,
        confidence_percentage: full.confidence_percentage,
        explanation: full.explanation?.summary ?? full.explanation ?? '',
        gradcam_heatmap: full.gradcam_heatmap?.heatmap_base64 ?? full.gradcam_heatmap ?? '',
        convnext_features_summary: full.convnext_features_summary,
        generator_attribution: full.generator_attribution ?? {
          attributed_family: 'Unknown',
          confidence: 0,
          family_probabilities: {},
          is_unseen_architecture: false,
          notes: '',
        },
        metadata_provenance: full.metadata_provenance ?? {
          has_exif: false,
          c2pa_manifest_found: false,
          c2pa_claim_summary: null,
          exif_details: {},
          authenticity_signals: {},
        },
        timestamp: full.created_at ?? undefined,
      };
      onLoadResult(result);
    } catch (e: any) {
      setError(`Could not load analysis: ${e.message}`);
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4 pb-4 border-b border-white/5">
        <div>
          <div className="flex items-center gap-2.5">
            <History className="w-5 h-5 text-[#96E071]" />
            <h2 className="text-xl font-bold text-white tracking-tight">Analysis History</h2>
          </div>
          <p className="font-mono text-xs text-gray-400 mt-1.5 flex items-center gap-1.5">
            <Database className="w-3 h-3" />
            Powered by Supabase — all scan results persist in the cloud
          </p>
        </div>
        <button
          onClick={fetchHistory}
          disabled={loading || !backendOnline}
          id="refresh-history-btn"
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#1C1F2B] hover:bg-[#25293A] text-[#96E071] border border-[#96E071]/30 hover:border-[#96E071] font-mono text-xs font-semibold transition-all disabled:opacity-40"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Backend offline warning */}
      {!backendOnline && (
        <div className="flex items-center gap-3 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 font-mono text-sm">
          <ServerCrash className="w-5 h-5 shrink-0" />
          <span>Backend is offline — connect the FastAPI server to view history.</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 font-mono text-sm">
          <ShieldAlert className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-[#111319] rounded-xl animate-pulse border border-white/5" />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && records.length === 0 && backendOnline && (
        <div className="flex flex-col items-center justify-center py-20 gap-4 text-gray-500">
          <Inbox className="w-12 h-12 opacity-30" />
          <p className="font-mono text-sm">No analyses yet — upload an image to get started!</p>
        </div>
      )}

      {/* Records list */}
      {!loading && records.length > 0 && (
        <div className="space-y-2">
          {records.map((r) => (
            <div
              key={r.id}
              className="group flex items-center gap-4 px-4 py-3.5 bg-[#111319] hover:bg-[#161A25] border border-[#2A2E3B] hover:border-[#96E071]/30 rounded-xl transition-all cursor-pointer"
              onClick={() => handleLoad(r.id)}
              id={`history-item-${r.id}`}
            >
              {/* Icon */}
              <div className={`shrink-0 w-9 h-9 rounded-lg flex items-center justify-center ${r.is_ai_generated ? 'bg-red-500/15 text-red-400' : 'bg-[#96E071]/15 text-[#96E071]'}`}>
                {r.is_ai_generated
                  ? <ShieldAlert className="w-4.5 h-4.5" />
                  : <ShieldCheck className="w-4.5 h-4.5" />
                }
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white truncate">{r.image_name}</p>
                <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                  <span className={`font-mono text-xs ${r.is_ai_generated ? 'text-red-400' : 'text-[#96E071]'}`}>
                    {r.prediction}
                  </span>
                  <span className="text-gray-600">·</span>
                  <span className="flex items-center gap-1 font-mono text-xs text-gray-500">
                    <Clock className="w-3 h-3" />
                    {formatDate(r.created_at)}
                  </span>
                </div>
              </div>

              {/* Confidence */}
              <div className="shrink-0">
                <ConfidenceBadge value={r.calibrated_confidence} isAi={r.is_ai_generated} />
              </div>

              {/* Arrow / loader */}
              <div className="shrink-0 text-gray-600 group-hover:text-[#96E071] transition-colors">
                {loadingId === r.id
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <ChevronRight className="w-4 h-4" />
                }
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer count */}
      {records.length > 0 && (
        <p className="text-center font-mono text-xs text-gray-600">
          Showing {records.length} most recent scans from Supabase
        </p>
      )}
    </div>
  );
};
