import React from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

interface DisclaimerBannerProps {
  compact?: boolean;
  className?: string;
}

export const DisclaimerBanner: React.FC<DisclaimerBannerProps> = ({ compact = false, className = '' }) => {
  if (compact) {
    return (
      <div
        role="alert"
        className={`w-full bg-amber-50/90 border border-amber-200/90 rounded-xl p-3 text-amber-950 flex items-center gap-3 text-xs shadow-2xs ${className}`}
      >
        <div className="p-1.5 bg-amber-100 rounded-lg text-amber-700 flex-shrink-0">
          <AlertTriangle className="w-3.5 h-3.5" />
        </div>
        <div className="leading-relaxed flex-1 min-w-0">
          <span className="font-semibold text-amber-900 mr-1.5 uppercase tracking-wider text-[10px] bg-amber-100/80 px-1.5 py-0.5 rounded">
            Clinical Evaluation Notice
          </span>
          <span className="text-amber-800">
            Educational risk-estimation tool only. Not a medical diagnostic device. Do not use to initiate or alter therapy. Clinical judgment takes precedence over algorithmic outputs.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      role="alert"
      className={`p-4 bg-amber-50/90 border border-amber-200/90 rounded-xl text-amber-950 flex items-start gap-3.5 text-sm shadow-xs ${className}`}
    >
      <div className="p-2 bg-amber-100 rounded-xl text-amber-700 flex-shrink-0 mt-0.5">
        <ShieldAlert className="w-4 h-4" />
      </div>
      <div className="space-y-1">
        <h4 className="font-semibold text-amber-900 text-xs uppercase tracking-wider flex items-center gap-1.5">
          <span>Clinical Notice & Medical Disclaimer</span>
          <span className="text-[10px] font-normal normal-case text-amber-700 bg-amber-100/80 px-1.5 py-0.5 rounded">
            CDSS Regulatory Guidance
          </span>
        </h4>
        <p className="text-amber-800 text-xs sm:text-sm leading-relaxed">
          Educational cardiovascular risk-estimation tool only. Not certified for clinical diagnosis. Do not use this result to start, stop, or change patient treatment. Always consult a qualified medical professional.
        </p>
      </div>
    </div>
  );
};
