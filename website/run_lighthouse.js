import { execSync } from 'child_process';
import fs from 'fs';

console.log('Running Lighthouse...');
try {
  execSync('npx lighthouse http://localhost:4173/Sparse2Full_Official/ --output json --output-path ./lh-report.json --chrome-flags="--headless --no-sandbox" --quiet', { stdio: 'inherit' });
  console.log('Lighthouse completed successfully.');
} catch (e) {
  console.log('Lighthouse encountered an error:', e.message);
}

try {
  const report = JSON.parse(fs.readFileSync('./lh-report.json', 'utf8'));
  console.log('\\n--- Lighthouse Performance Report ---');
  console.log('Performance Score:', report.categories.performance.score * 100);
  console.log('First Contentful Paint (FCP):', report.audits['first-contentful-paint'].displayValue);
  console.log('Largest Contentful Paint (LCP):', report.audits['largest-contentful-paint'].displayValue);
  console.log('Time to Interactive (TTI):', report.audits['interactive'].displayValue);
  console.log('Total Blocking Time (TBT):', report.audits['total-blocking-time'].displayValue);
  console.log('Speed Index:', report.audits['speed-index'].displayValue);
  
  console.log('\\n--- Accessibility & Best Practices ---');
  console.log('Accessibility Score:', report.categories.accessibility.score * 100);
  console.log('Best Practices Score:', report.categories['best-practices'].score * 100);
  console.log('SEO Score:', report.categories.seo.score * 100);

  const errors = report.audits['errors-in-console'];
  console.log('\\n--- Console Errors ---');
  if (errors && errors.details && errors.details.items.length > 0) {
    errors.details.items.forEach(item => {
      console.log(`[${item.source}] ${item.description}`);
    });
  } else {
    console.log('No console errors found.');
  }
} catch (e) {
  console.log('Failed to parse report:', e.message);
}
