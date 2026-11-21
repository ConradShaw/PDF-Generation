/** The Shaw Strengths Matrix chart (cognitive x temporal preferences), shared by the reports that print it. */
export function strengthsMatrixHtml(): string {
  return `
    <table class="matrix">
      <tr><td colspan="5">Shaw Strengths Matrix®</td></tr>
      <tr><td colspan="2" rowspan="2">Character Strengths</td><td colspan="3">Temporal Preferences</td></tr>
      <tr><td>Past Reflection</td><td>Present Engagement</td><td>Future Anticipation</td></tr>
      <tr><td rowspan="4">Cognitive Preferences</td><td>Intuition</td><td class="cell-blue">Foresight</td><td class="cell-blue">Confidence</td><td class="cell-blue">Initiative</td></tr>
      <tr><td>Thinking</td><td class="cell-blue">Curiosity</td><td class="cell-blue">Objectivity</td><td class="cell-blue">Fairness</td></tr>
      <tr><td>Feeling</td><td class="cell-blue">Empathy</td><td class="cell-blue">Tenacity</td><td class="cell-blue">Prudence</td></tr>
      <tr><td>Sensing</td><td class="cell-blue">Discernment</td><td class="cell-blue">Practicality</td><td class="cell-blue">Discipline</td></tr>
    </table>`;
}
