import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  UploadCloud, 
  FileVideo, 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  Sparkles, 
  Eye, 
  Download, 
  RefreshCw,
  AlertTriangle,
  Check,
  X
} from 'lucide-react';
import { api } from '../api/client';

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [confidenceCut, setConfidenceCut] = useState(0.30);

  // Synchronous photo detection result
  const [result, setResult] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);
  const [filterMode, setFilterMode] = useState('all'); // 'all' | 'violations' | 'safe'

  // Background batch jobs list
  const [jobs, setJobs] = useState([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);

  const fetchJobs = async () => {
    try {
      const res = await api.getJobs();
      setJobs(res.data || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setErrorMsg('');
    setResult(null);

    const ext = selectedFile.name.toLowerCase();
    if (ext.endsWith('.jpg') || ext.endsWith('.jpeg') || ext.endsWith('.png') || ext.endsWith('.webp')) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  // Run AI Detection on the single photo
  const handleDetectPhoto = async (e) => {
    if (e) e.preventDefault();
    if (!file) return;

    setErrorMsg('');
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', confidenceCut.toString());

    try {
      const res = await api.detectPhoto(formData);
      if (res.data && res.data.success) {
        setResult(res.data);
        fetchJobs();
      } else {
        throw new Error(res.data?.detail || 'Detection failed');
      }
    } catch (err) {
      console.error('[UploadPage] Detection error:', err);
      const status = err.response?.status;
      const detail = err.response?.data?.detail || err.response?.data?.message;
      if (status === 400) {
        setErrorMsg(detail || 'Unsupported file type. Please upload a .jpg or .png image.');
      } else if (status === 500) {
        setErrorMsg('Server error during detection. Please try again. If the problem persists, check the console.');
      } else if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setErrorMsg('Detection is taking too long. Please try a smaller image.');
      } else {
        setErrorMsg(detail || err.message || 'Detection failed. Please check the image and try again.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Quick test sample loader
  const loadSample = async (samplePath, filename) => {
    setErrorMsg('');
    setResult(null);
    setIsProcessing(true);
    try {
      const resp = await fetch(api.getEvidenceUrl(samplePath));
      const blob = await resp.blob();
      const sampleFile = new File([blob], filename, { type: blob.type || 'image/jpeg' });
      handleFileChange(sampleFile);

      const formData = new FormData();
      formData.append('file', sampleFile);
      formData.append('conf', confidenceCut.toString());
      const res = await api.detectPhoto(formData);
      setResult(res.data);
    } catch (err) {
      setErrorMsg(`Could not load sample: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const filteredDetections = (result?.detections || []).filter((d) => {
    if (filterMode === 'violations') return d.status === 'violation' || d.status === 'moving' || d.status === 'unverified';
    if (filterMode === 'safe') return d.status === 'safe';
    return true;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <span>🏍️ Two-Wheeler Parking & Violation Detector</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Upload <strong>ANY bike parking photo</strong> to detect all vehicles simultaneously. The AI evaluates proper alignment and highlights all unpositioned / illegally parked bikes together in <strong>ONE single annotated photo</strong>.
        </p>
      </div>

      {/* Upload & Config Card */}
      <div className="bg-slate-800/80 border border-slate-700/80 rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pb-3 border-b border-slate-700/60">
          <div className="text-xs text-slate-300 font-semibold flex items-center gap-2">
            <span>Detection Sensitivity (Min Accuracy Cut):</span>
            <span className="px-2 py-0.5 rounded bg-sky-950 text-sky-400 font-mono font-bold border border-sky-800">
              {(confidenceCut * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-slate-400">Low (30%)</span>
            <input
              type="range"
              min="0.25"
              max="0.80"
              step="0.05"
              value={confidenceCut}
              onChange={(e) => setConfidenceCut(parseFloat(e.target.value))}
              className="w-32 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-sky-500"
            />
            <span className="text-[11px] text-slate-400">Strict (80%)</span>
          </div>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files?.[0]) handleFileChange(e.dataTransfer.files[0]);
          }}
          className={`border-2 border-dashed rounded-xl p-6 text-center flex flex-col items-center justify-center cursor-pointer relative transition-all ${
            file
              ? 'border-sky-500 bg-sky-950/20'
              : 'border-slate-700 hover:border-slate-600 bg-slate-900/40'
          }`}
        >
          <input
            id="photo-upload-input"
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.mp4"
            onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
          />
          <UploadCloud className={`w-10 h-10 mb-2 ${file ? 'text-sky-400' : 'text-slate-400'}`} />
          <p className="text-sm font-semibold text-white">
            {file ? file.name : 'Click or Drag & Drop ANY bike parking photo here'}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Accepts JPG, PNG, WEBP images from CCTV, mobile, or parking lots
          </p>
          {file && (
            <span className="mt-2 text-xs text-sky-400 font-mono bg-sky-950/80 px-2.5 py-0.5 rounded-full border border-sky-800">
              Selected: {(file.size / 1024).toFixed(1)} KB
            </span>
          )}
        </div>

        {/* Action Button & Sample Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
          <div className="flex items-center gap-2 text-xs text-slate-400 flex-wrap">
            <span>Quick Test Samples:</span>
            <button
              type="button"
              onClick={() => loadSample('6203aa8a-d955-462d-b663-a6c22101968c.jpg', 'parking_lot_sample.jpg')}
              className="px-2.5 py-1 rounded bg-slate-700/80 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-600"
            >
              📸 Parking Lot (Colab Model)
            </button>
            <button
              type="button"
              onClick={() => loadSample('254fab24-dfcf-4e81-877f-d2b75f58c3a8.jpg', 'street_motorcycles.jpg')}
              className="px-2.5 py-1 rounded bg-slate-700/80 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-600"
            >
              🏍️ Street Bikes (Outlier Test)
            </button>
          </div>

          <button
            type="button"
            onClick={handleDetectPhoto}
            disabled={!file || isProcessing}
            className="w-full sm:w-auto flex items-center justify-center space-x-2 px-6 py-2.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg shadow-md transition-all"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Analyzing Parking & Violations...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Detect Parking Violations Now</span>
              </>
            )}
          </button>
        </div>

        {errorMsg && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs rounded-lg flex items-center space-x-2">
            <AlertCircle className="w-4 h-4" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* ────────────────── RESULTS SECTION (SHOWN PROMINENTLY) ────────────────── */}
      {result && (
        <div className="bg-slate-900 border border-slate-700/80 rounded-2xl p-6 shadow-xl space-y-6">
          {/* Header & Metrics Banner */}
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <span className="text-xs text-sky-400 uppercase tracking-wider font-bold">Detection Result</span>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span>Violations Highlighted on Output Photo</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Red = Unpositioned / Violations, Amber = Misaligned / Risk. Correctly parked bikes are left clean.
              </p>
            </div>

            {/* Badges */}
            <div className="flex items-center gap-2 flex-wrap">
              <div className="px-3 py-1.5 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-400 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
                <span>{result.violations_count} Violations</span>
              </div>
              <div className="px-3 py-1.5 rounded-xl bg-emerald-950/80 border border-emerald-800 text-emerald-400 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>{result.safe_count} Legal Parking</span>
              </div>
              {(result.moving_count > 0) && (
                <div className="px-3 py-1.5 rounded-xl bg-slate-700/80 border border-slate-600 text-slate-300 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                  <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                  <span>{result.moving_count} Moving (Rider)</span>
                </div>
              )}
              {(result.unverified_count > 0) && (
                <div className="px-3 py-1.5 rounded-xl bg-cyan-950/60 border border-cyan-800/60 text-cyan-300 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                  <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                  <span>{result.unverified_count} Unverified</span>
                </div>
              )}
              {result.vacant_count > 0 && (
                <div className="px-3 py-1.5 rounded-xl bg-cyan-950/80 border border-cyan-800 text-cyan-400 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                  <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                  <span>{result.vacant_count} Empty Slots</span>
                </div>
              )}
              <div className="px-3 py-1.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 text-xs font-bold">
                Total: {result.total_detected}
              </div>
              {result.violations_count > 0 && (
                <Link
                  to="/violations"
                  className="px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold flex items-center gap-1 shadow-sm transition-colors"
                >
                  <span>View in Violations Section →</span>
                </Link>
              )}
            </div>
          </div>

          {/* Controls Bar */}
          <div className="flex items-center justify-between flex-wrap gap-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-400 font-medium">View Mode:</span>
              <button
                type="button"
                onClick={() => setShowOriginal(false)}
                className={`px-3 py-1 rounded-lg font-semibold border ${
                  !showOriginal
                    ? 'bg-sky-600 text-white border-sky-500'
                    : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-white'
                }`}
              >
                Annotated Photo (All Boxes)
              </button>
              <button
                type="button"
                onClick={() => setShowOriginal(true)}
                className={`px-3 py-1 rounded-lg font-semibold border ${
                  showOriginal
                    ? 'bg-sky-600 text-white border-sky-500'
                    : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-white'
                }`}
              >
                Original Photo
              </button>
            </div>

            <a
              href={result.annotated_image_url}
              download={`${result.filename}_annotated.jpg`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Save Image</span>
            </a>
          </div>

          {/* ── THE MASTER ANNOTATED IMAGE ── */}
          <div className="rounded-xl overflow-hidden border border-slate-800 bg-black flex items-center justify-center relative shadow-inner">
            <img
              src={showOriginal ? result.original_image_url : result.annotated_image_url}
              alt="Detection Master View"
              className="w-full max-h-[75vh] object-contain"
            />
          </div>

          {/* ── DETECTION BOX INSPECTOR (violations + moving + unverified) ── */}
          {(result.detections || []).some(d => d.status !== 'safe') && (
            <div className="bg-slate-950/60 border border-slate-700/60 rounded-xl p-4 space-y-3">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <span className="p-1 rounded-md bg-rose-600/20 text-rose-400 font-bold">🔍</span>
                  <h3 className="text-sm font-bold text-white">
                    Detection Box Inspector
                    {result.violations_count > 0 && <span className="ml-2 text-rose-400">({result.violations_count} violations)</span>}
                    {(result.moving_count > 0) && <span className="ml-2 text-slate-400">({result.moving_count} moving)</span>}
                    {(result.unverified_count > 0) && <span className="ml-2 text-cyan-400">({result.unverified_count} unverified)</span>}
                  </h3>
                </div>
                <span className="text-[11px] text-slate-400">
                  🔴 Violation &nbsp;|&nbsp; ⚪ Moving (Rider) &nbsp;|&nbsp; 🟡 Unverified &nbsp;|&nbsp; 🟠 Risk
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 pt-1">
                {(result.detections || [])
                  .filter(d => d.status !== 'safe')
                  .map((d) => {
                    const borderColor =
                      d.status === 'violation' ? 'border-rose-800/50 hover:border-rose-500' :
                      d.status === 'moving'    ? 'border-slate-600/50 hover:border-slate-400' :
                      d.status === 'unverified'? 'border-cyan-800/40 hover:border-cyan-500' :
                                                 'border-amber-800/40 hover:border-amber-500';
                    const badgeBg =
                      d.status === 'violation' ? 'bg-rose-600' :
                      d.status === 'moving'    ? 'bg-slate-500' :
                      d.status === 'unverified'? 'bg-cyan-700' :
                                                 'bg-amber-600';
                    const badgeLabel =
                      d.status === 'violation' ? '🔴 VIOLATION' :
                      d.status === 'moving'    ? '⚪ MOVING' :
                      d.status === 'unverified'? '🟡 UNVERIFIED' :
                                                 '🟠 RISK';
                    const reasonColor =
                      d.status === 'violation' ? 'text-rose-300' :
                      d.status === 'moving'    ? 'text-slate-400' :
                      d.status === 'unverified'? 'text-cyan-400' :
                                                 'text-amber-300';
                    return (
                      <div
                        key={d.index}
                        className={`bg-slate-900/90 border ${borderColor} rounded-xl p-3 flex flex-col justify-between transition-all group shadow-sm`}
                      >
                        <div className="rounded-lg overflow-hidden border border-slate-800 bg-black aspect-video flex items-center justify-center mb-2.5 relative">
                          {d.crop_url ? (
                            <img
                              src={d.crop_url}
                              alt={`Box #${d.index}`}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                            />
                          ) : (
                            <span className="text-slate-600 text-xs">No crop</span>
                          )}
                          <span className={`absolute top-1.5 left-1.5 px-2 py-0.5 rounded text-[10px] font-bold ${badgeBg} text-white shadow`}>
                            #{d.index} {badgeLabel}
                          </span>
                        </div>

                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-bold text-white capitalize">{d.class_name}</span>
                            <span className="text-[10px] font-mono font-bold text-sky-400 bg-sky-950 px-1.5 py-0.5 rounded border border-sky-800">
                              {d.confidence_percent}
                            </span>
                          </div>
                          <p className={`text-[11px] ${reasonColor} font-medium leading-tight`}>
                            {d.reason}
                          </p>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Filter Pills */}
          <div className="flex items-center justify-between flex-wrap gap-3 pt-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Filter list:</span>
              <button
                onClick={() => setFilterMode('all')}
                className={`px-2.5 py-1 text-xs rounded-lg font-semibold ${
                  filterMode === 'all' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                All ({result.detections?.length || 0})
              </button>
              <button
                onClick={() => setFilterMode('violations')}
                className={`px-2.5 py-1 text-xs rounded-lg font-semibold ${
                  filterMode === 'violations' ? 'bg-rose-900/60 text-rose-300 border border-rose-800' : 'text-slate-400 hover:text-rose-400'
                }`}
              >
                Violations Only ({result.violations_count})
              </button>
              <button
                onClick={() => setFilterMode('safe')}
                className={`px-2.5 py-1 text-xs rounded-lg font-semibold ${
                  filterMode === 'safe' ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-800' : 'text-slate-400 hover:text-emerald-400'
                }`}
              >
                Legal Parking ({result.safe_count})
              </button>
            </div>
          </div>

          {/* Table of all detected vehicles */}
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Box #</th>
                  <th className="py-2.5 px-3">Preview</th>
                  <th className="py-2.5 px-3">Parking Status</th>
                  <th className="py-2.5 px-3">Class</th>
                  <th className="py-2.5 px-3">Confidence</th>
                  <th className="py-2.5 px-3">Violation Reason / Positioning</th>
                  <th className="py-2.5 px-3 text-right">Bounding Box</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredDetections.map((det) => {
                  const isVio = det.status === 'violation';
                  const isSafe = det.status === 'safe';
                  const isVacant = det.status === 'vacant_slot';
                  const isMoving = det.status === 'moving';
                  const isUnverified = det.status === 'unverified';
                  return (
                    <tr
                       key={det.index}
                       className={isVio ? 'bg-rose-950/20 hover:bg-rose-950/30' : isMoving ? 'bg-slate-800/20 hover:bg-slate-800/30' : 'hover:bg-slate-800/30'}
                    >
                       <td className="py-2.5 px-3 font-bold text-white">#{det.index}</td>
                       <td className="py-2 px-3">
                         {det.crop_url ? (
                           <img
                             src={det.crop_url}
                             alt={`Box #${det.index}`}
                             className="w-12 h-9 rounded object-cover border border-slate-700 bg-black"
                           />
                         ) : (
                           <span className="text-slate-600 font-mono text-[10px]">—</span>
                         )}
                       </td>
                       <td className="py-2.5 px-3">
                         {isVio && (
                           <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40">
                             <X className="w-3 h-3" />
                             <span>[X] VIOLATION</span>
                           </span>
                         )}
                         {isSafe && (
                           <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                             <Check className="w-3 h-3" />
                             <span>[OK] LEGAL PARKING</span>
                           </span>
                         )}
                         {isVacant && (
                           <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                             <span>[O] EMPTY SLOT</span>
                           </span>
                         )}
                         {det.status === 'risk' && (
                           <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/40">
                             <AlertTriangle className="w-3 h-3" />
                             <span>[!] RISK (MISALIGNED)</span>
                           </span>
                         )}
                         {isMoving && (
                           <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-slate-500/20 text-slate-300 border border-slate-500/40">
                             <span>⚪ MOVING (RIDER)</span>
                           </span>
                         )}
                         {isUnverified && (
                           <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-cyan-900/40 text-cyan-300 border border-cyan-700/40">
                             <span>🟡 UNVERIFIED</span>
                           </span>
                         )}
                       </td>
                       <td className="py-2.5 px-3 capitalize font-medium text-slate-200">
                         {det.class_name}
                       </td>
                       <td className="py-2.5 px-3 font-mono font-semibold text-sky-400">
                         {det.confidence_percent}
                       </td>
                       <td className={`py-2.5 px-3 font-medium ${isVio ? 'text-rose-300 font-semibold' : isMoving ? 'text-slate-400' : isUnverified ? 'text-cyan-400' : 'text-slate-400'}`}>
                         {det.reason || '-'}
                       </td>
                       <td className="py-2.5 px-3 text-right font-mono text-[11px] text-slate-400">
                         [{det.bbox?.join(', ')}]
                       </td>
                     </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ────────────────── BATCH QUEUE & VIDEO FOOTAGE ────────────────── */}
      <div className="bg-slate-800/80 border border-slate-700/80 rounded-xl overflow-hidden shadow-sm space-y-3 p-5">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-white text-sm">Background CCTV & Video Queue</h2>
          <span className="text-[11px] text-slate-400">Long footage batch processing history</span>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-700">
              <tr>
                <th className="py-3 px-4 font-semibold">File</th>
                <th className="py-3 px-4 font-semibold">Status</th>
                <th className="py-3 px-4 font-semibold">Progress Message</th>
                <th className="py-3 px-4 font-semibold">Submitted</th>
                <th className="py-3 px-4 font-semibold text-right">Summary</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/60">
              {jobs.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-8 text-center text-slate-500">
                    No background video jobs recorded yet.
                  </td>
                </tr>
              ) : (
                jobs.map((j) => (
                  <tr key={j.job_id} className="hover:bg-slate-700/20">
                    <td className="py-3 px-4 font-medium text-white flex items-center space-x-2">
                      <FileVideo className="w-4 h-4 text-sky-400" />
                      <span>{j.filename}</span>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold capitalize ${
                          j.status === 'completed'
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                            : j.status === 'failed'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                        }`}
                      >
                        {j.status === 'processing' && <Loader2 className="w-3 h-3 animate-spin mr-1" />}
                        {j.status === 'completed' && <CheckCircle className="w-3 h-3 mr-1" />}
                        <span>{j.status}</span>
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400">{j.message}</td>
                    <td className="py-3 px-4 text-slate-400 font-mono">
                      {new Date(j.created_at).toLocaleTimeString()}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-xs">
                      {j.result_summary ? (
                        <div className="flex flex-col items-end space-y-1">
                          <span className="text-emerald-400 font-bold">
                            {j.result_summary.detections !== undefined
                              ? `${j.result_summary.detections} detected`
                              : `${j.result_summary.parking_events} events`}
                          </span>
                          {j.result_summary.master_snapshot && (
                            <button
                              type="button"
                              onClick={() => setSelectedSnapshot(api.getEvidenceUrl(j.result_summary.master_snapshot))}
                              className="text-[11px] text-sky-400 hover:text-sky-300 underline font-sans"
                            >
                              👁️ View Result Photo
                            </button>
                          )}
                        </div>
                      ) : (
                        '-'
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Snapshot Modal for background jobs */}
      {selectedSnapshot && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-white text-sm">AI Detection Master Image</h3>
                <p className="text-xs text-slate-400">All vehicles & parking slots marked together in one image</p>
              </div>
              <button
                onClick={() => setSelectedSnapshot(null)}
                className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded-lg font-bold"
              >
                ✕ Close
              </button>
            </div>
            <div className="p-4 flex flex-col items-center justify-center bg-slate-950">
              <img
                src={selectedSnapshot}
                alt="Detection Evidence"
                className="max-h-[70vh] rounded-lg border border-slate-800 object-contain shadow-lg"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
