import React, { useState } from 'react';
import type { PatientInput } from '../../types/api';
import { RefreshCw, Stethoscope, User, Activity, FileSpreadsheet } from 'lucide-react';

interface PatientFormProps {
  onSubmit: (data: PatientInput) => Promise<void>;
  loading: boolean;
  onExplainRequested?: (data: PatientInput) => void;
}

const PRESET_HIGH_RISK: PatientInput = {
  age: 63.0,
  sex: 1.0,
  cp: 4.0,           // Asymptomatic
  trestbps: 160.0,
  chol: 286.0,
  fbs: 1.0,
  restecg: 2.0,       // LV hypertrophy
  thalach: 108.0,
  exang: 1.0,         // Exercise angina
  oldpeak: 3.2,
  slope: 2.0,         // Flat
  ca: 3.0,            // 3 vessels
  thal: 7.0,          // Reversible defect
};

const PRESET_LOW_RISK: PatientInput = {
  age: 37.0,
  sex: 0.0,
  cp: 3.0,           // Non-anginal
  trestbps: 115.0,
  chol: 190.0,
  fbs: 0.0,
  restecg: 0.0,       // Normal
  thalach: 178.0,
  exang: 0.0,         // No angina
  oldpeak: 0.0,
  slope: 1.0,         // Upsloping
  ca: 0.0,            // 0 vessels
  thal: 3.0,          // Normal
};

export const PatientForm: React.FC<PatientFormProps> = ({
  onSubmit,
  loading,
}) => {
  const [formData, setFormData] = useState<PatientInput>({ ...PRESET_HIGH_RISK });
  const [caUnknown, setCaUnknown] = useState<boolean>(false);
  const [thalUnknown, setThalUnknown] = useState<boolean>(false);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const errors: Record<string, string> = {};

    if (formData.age < 18 || formData.age > 120) {
      errors.age = 'Age must be between 18 and 120 years.';
    }
    if (formData.trestbps < 60 || formData.trestbps > 250) {
      errors.trestbps = 'Resting blood pressure must be between 60 and 250 mm Hg.';
    }
    if (formData.chol < 80 || formData.chol > 600) {
      errors.chol = 'Serum cholesterol must be between 80 and 600 mg/dl.';
    }
    if (formData.thalach < 50 || formData.thalach > 230) {
      errors.thalach = 'Max heart rate must be between 50 and 230 bpm.';
    }
    if (formData.oldpeak < 0.0 || formData.oldpeak > 10.0) {
      errors.oldpeak = 'ST depression must be between 0.0 and 10.0 mm.';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    const payload: PatientInput = {
      ...formData,
      ca: caUnknown ? null : formData.ca,
      thal: thalUnknown ? null : formData.thal,
    };

    onSubmit(payload);
  };

  const handlePreset = (preset: PatientInput) => {
    setFormData({ ...preset });
    setCaUnknown(preset.ca === null);
    setThalUnknown(preset.thal === null);
    setValidationErrors({});
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
      {/* Form Header & Presets */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-teal-600" />
            Patient Clinical Characteristics
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            13 standard cardiac variables used by the trained ML pipeline.
          </p>
        </div>

        {/* Preset Selector */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-400 font-medium hidden sm:inline">Presets:</span>
          <button
            type="button"
            onClick={() => handlePreset(PRESET_HIGH_RISK)}
            className="text-xs px-2.5 py-1 rounded-lg bg-rose-50 text-rose-800 border border-rose-200 hover:bg-rose-100 font-medium transition-all"
          >
            High Risk Sample
          </button>
          <button
            type="button"
            onClick={() => handlePreset(PRESET_LOW_RISK)}
            className="text-xs px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 hover:bg-emerald-100 font-medium transition-all"
          >
            Low Risk Sample
          </button>
        </div>
      </div>

      {/* SECTION 1: Patient Demographics & Baseline */}
      <div className="space-y-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5 border-b border-slate-100 pb-1.5">
          <User className="w-3.5 h-3.5 text-teal-600" />
          1. Patient Demographics & Baseline
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Age */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Age (Years) <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min={18}
              max={120}
              step={1}
              value={formData.age}
              onChange={(e) => setFormData({ ...formData, age: parseFloat(e.target.value) || 0 })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                validationErrors.age ? 'border-rose-400 bg-rose-50/30' : 'border-slate-300 bg-white'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            />
            {validationErrors.age && <p className="text-[11px] text-rose-600 mt-1">{validationErrors.age}</p>}
            <p className="text-[11px] text-slate-400 mt-0.5">Permissible range: 18 – 120 years</p>
          </div>

          {/* Sex */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Biological Sex <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.sex}
              onChange={(e) => setFormData({ ...formData, sex: parseFloat(e.target.value) as 0 | 1 })}
              className="w-full px-3 py-2 text-sm rounded-xl border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600"
            >
              <option value={1}>Male (1.0)</option>
              <option value={0}>Female (0.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Biological sex at birth</p>
          </div>
        </div>
      </div>

      {/* SECTION 2: Clinical Vitals & Lab Measurements */}
      <div className="space-y-4 pt-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5 border-b border-slate-100 pb-1.5">
          <Activity className="w-3.5 h-3.5 text-teal-600" />
          2. Clinical Vitals & Laboratory Biomarkers
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Trestbps */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Resting BP (mm Hg) <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min={60}
              max={250}
              step={1}
              value={formData.trestbps}
              onChange={(e) => setFormData({ ...formData, trestbps: parseFloat(e.target.value) || 0 })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                validationErrors.trestbps ? 'border-rose-400 bg-rose-50/30' : 'border-slate-300 bg-white'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            />
            {validationErrors.trestbps && <p className="text-[11px] text-rose-600 mt-1">{validationErrors.trestbps}</p>}
            <p className="text-[11px] text-slate-400 mt-0.5">Systolic on admission (60–250)</p>
          </div>

          {/* Chol */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Serum Cholesterol (mg/dl) <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min={80}
              max={600}
              step={1}
              value={formData.chol}
              onChange={(e) => setFormData({ ...formData, chol: parseFloat(e.target.value) || 0 })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                validationErrors.chol ? 'border-rose-400 bg-rose-50/30' : 'border-slate-300 bg-white'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            />
            {validationErrors.chol && <p className="text-[11px] text-rose-600 mt-1">{validationErrors.chol}</p>}
            <p className="text-[11px] text-slate-400 mt-0.5">Total cholesterol (80–600)</p>
          </div>

          {/* FBS */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Fasting Blood Sugar <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.fbs}
              onChange={(e) => setFormData({ ...formData, fbs: parseFloat(e.target.value) as 0 | 1 })}
              className="w-full px-3 py-2 text-sm rounded-xl border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600"
            >
              <option value={0}>Normal &le; 120 mg/dl (0.0)</option>
              <option value={1}>Elevated &gt; 120 mg/dl (1.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Indicator for diabetes / hyperglycemia</p>
          </div>

          {/* RestECG */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Resting ECG Results <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.restecg}
              onChange={(e) => setFormData({ ...formData, restecg: parseFloat(e.target.value) as 0 | 1 | 2 })}
              className="w-full px-3 py-2 text-sm rounded-xl border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600"
            >
              <option value={0}>Normal (0.0)</option>
              <option value={1}>ST-T Wave Abnormality (1.0)</option>
              <option value={2}>Left Ventricular Hypertrophy (2.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Electrocardiographic rest state</p>
          </div>

          {/* Thalach */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Max Heart Rate (bpm) <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min={50}
              max={230}
              step={1}
              value={formData.thalach}
              onChange={(e) => setFormData({ ...formData, thalach: parseFloat(e.target.value) || 0 })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                validationErrors.thalach ? 'border-rose-400 bg-rose-50/30' : 'border-slate-300 bg-white'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            />
            {validationErrors.thalach && <p className="text-[11px] text-rose-600 mt-1">{validationErrors.thalach}</p>}
            <p className="text-[11px] text-slate-400 mt-0.5">Max exercise heart rate (50–230)</p>
          </div>

          {/* Exang */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Exercise-Induced Angina <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.exang}
              onChange={(e) => setFormData({ ...formData, exang: parseFloat(e.target.value) as 0 | 1 })}
              className="w-full px-3 py-2 text-sm rounded-xl border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600"
            >
              <option value={0}>No (0.0)</option>
              <option value={1}>Yes (1.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Chest pain triggered by exertion</p>
          </div>
        </div>
      </div>

      {/* SECTION 3: Symptom Presentation & Stress Test Diagnostics */}
      <div className="space-y-4 pt-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5 border-b border-slate-100 pb-1.5">
          <FileSpreadsheet className="w-3.5 h-3.5 text-teal-600" />
          3. Symptom Classification & Diagnostic Tests
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {/* CP */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Chest Pain Type (cp) <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.cp}
              onChange={(e) => setFormData({ ...formData, cp: parseFloat(e.target.value) as 1 | 2 | 3 | 4 })}
              className="w-full px-3 py-2 text-sm rounded-xl border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600"
            >
              <option value={1}>Typical Angina (1.0)</option>
              <option value={2}>Atypical Angina (2.0)</option>
              <option value={3}>Non-Anginal Pain (3.0)</option>
              <option value={4}>Asymptomatic (4.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Clinical chest pain classification</p>
          </div>

          {/* Oldpeak */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              ST Depression (Oldpeak) <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min={0.0}
              max={10.0}
              step={0.1}
              value={formData.oldpeak}
              onChange={(e) => setFormData({ ...formData, oldpeak: parseFloat(e.target.value) || 0 })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                validationErrors.oldpeak ? 'border-rose-400 bg-rose-50/30' : 'border-slate-300 bg-white'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            />
            {validationErrors.oldpeak && <p className="text-[11px] text-rose-600 mt-1">{validationErrors.oldpeak}</p>}
            <p className="text-[11px] text-slate-400 mt-0.5">Exercise vs rest ST depression in mm</p>
          </div>

          {/* Slope */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              ST Slope <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.slope}
              onChange={(e) => setFormData({ ...formData, slope: parseFloat(e.target.value) as 1 | 2 | 3 })}
              className="w-full px-3 py-2 text-sm rounded-xl border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600"
            >
              <option value={1}>Upsloping (1.0)</option>
              <option value={2}>Flat (2.0)</option>
              <option value={3}>Downsloping (3.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Slope of peak exercise ST segment</p>
          </div>

          {/* CA */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Major Fluoroscopy Vessels (ca)
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-teal-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={caUnknown}
                  onChange={(e) => setCaUnknown(e.target.checked)}
                  className="rounded text-teal-600 focus:ring-teal-500"
                />
                <span>Unknown (Median Imputed)</span>
              </label>
            </div>
            <select
              disabled={caUnknown}
              value={formData.ca ?? 0}
              onChange={(e) => setFormData({ ...formData, ca: parseFloat(e.target.value) })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                caUnknown ? 'bg-slate-100 text-slate-400 border-slate-200' : 'bg-white border-slate-300'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            >
              <option value={0}>0 Major Vessels (0.0)</option>
              <option value={1}>1 Major Vessel (1.0)</option>
              <option value={2}>2 Major Vessels (2.0)</option>
              <option value={3}>3 Major Vessels (3.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Vessels colored by fluoroscopy (0–3; median imputation if missing)</p>
          </div>

          {/* Thal */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Thallium Stress Defect (thal)
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-teal-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={thalUnknown}
                  onChange={(e) => setThalUnknown(e.target.checked)}
                  className="rounded text-teal-600 focus:ring-teal-500"
                />
                <span>Unknown (Most-Frequent Imputed)</span>
              </label>
            </div>
            <select
              disabled={thalUnknown}
              value={formData.thal ?? 3}
              onChange={(e) => setFormData({ ...formData, thal: parseFloat(e.target.value) })}
              className={`w-full px-3 py-2 text-sm rounded-xl border ${
                thalUnknown ? 'bg-slate-100 text-slate-400 border-slate-200' : 'bg-white border-slate-300'
              } focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600`}
            >
              <option value={3}>Normal Perfusion (3.0)</option>
              <option value={6}>Fixed Defect (6.0)</option>
              <option value={7}>Reversible Defect (7.0)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-0.5">Myocardial nuclear scintigraphy</p>
          </div>
        </div>
      </div>

      {/* Form Submission */}
      <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
        <p className="text-xs text-slate-500">
          All values are validated against the backend schema before submission.
        </p>

        <button
          type="submit"
          disabled={loading}
          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-teal-700 hover:bg-teal-800 text-white font-semibold text-sm transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-600/30 disabled:opacity-60"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Evaluating Risk...</span>
            </>
          ) : (
            <>
              <Stethoscope className="w-4 h-4" />
              <span>Compute Risk Assessment</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};
