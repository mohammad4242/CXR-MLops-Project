// Display metadata for the 18 thoracic findings returned by the
// "densenet121-res224-all" TorchXRayVision model. Keys match the raw pathology
// names from the API exactly; we map them to friendly labels and one-line,
// plain-language descriptions for tooltips.
export const PATHOLOGY_INFO = {
  Atelectasis: { label: 'Atelectasis', desc: 'Partial or complete collapse of a lung or a lobe.' },
  Cardiomegaly: { label: 'Cardiomegaly', desc: 'Enlarged heart silhouette on the radiograph.' },
  Consolidation: { label: 'Consolidation', desc: 'Air in the lungs replaced by fluid or solid material.' },
  Edema: { label: 'Edema', desc: 'Excess fluid accumulating in the lung tissue.' },
  Effusion: { label: 'Pleural Effusion', desc: 'Fluid collecting between the lung and the chest wall.' },
  Emphysema: { label: 'Emphysema', desc: 'Damaged air sacs causing trapped air in the lungs.' },
  Enlarged_Cardiomediastinum: { label: 'Enlarged Cardiomediastinum', desc: 'Widening of the central chest structures.' },
  Fibrosis: { label: 'Fibrosis', desc: 'Scarring and thickening of lung tissue.' },
  Fracture: { label: 'Fracture', desc: 'Possible rib or other bony fracture.' },
  Hernia: { label: 'Hernia', desc: 'Protrusion of an organ through the diaphragm.' },
  Infiltration: { label: 'Infiltration', desc: 'Substance denser than air spreading within the lungs.' },
  Lung_Lesion: { label: 'Lung Lesion', desc: 'A localized abnormal area of lung tissue.' },
  Lung_Opacity: { label: 'Lung Opacity', desc: 'A hazy area that may obscure underlying structures.' },
  Mass: { label: 'Mass', desc: 'A lesion larger than 3 cm across.' },
  Nodule: { label: 'Nodule', desc: 'A small rounded opacity up to 3 cm across.' },
  Pleural_Thickening: { label: 'Pleural Thickening', desc: 'Thickening of the lining around the lungs.' },
  Pneumonia: { label: 'Pneumonia', desc: 'Infection that inflames the lung air sacs.' },
  Pneumothorax: { label: 'Pneumothorax', desc: 'Air in the pleural space, which can collapse a lung.' },
}

// Confidence bands are a display-only heuristic used to colour the bars. Because
// only findings at/above 65% are ever shown, the thresholds are tuned for that
// range. These are NOT clinical severity levels.
export const SEVERITY_LABEL = { low: 'Elevated', moderate: 'High', high: 'Very high' }

export function severity(score) {
  if (score >= 0.85) return 'high'
  if (score >= 0.75) return 'moderate'
  return 'low'
}

/** Friendly label for a raw pathology key (falls back to de-underscored name). */
export function prettyName(key) {
  return PATHOLOGY_INFO[key]?.label ?? key.replaceAll('_', ' ')
}
