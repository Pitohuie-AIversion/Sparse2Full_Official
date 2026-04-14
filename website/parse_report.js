import fs from 'fs';

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

  // Network failures? (Check if any requests failed)
  console.log('\\n--- Network Requests Status ---');
  const networkRequests = report.audits['network-requests'];
  let allGood = true;
  if (networkRequests && networkRequests.details && networkRequests.details.items) {
    networkRequests.details.items.forEach(item => {
      if (item.statusCode >= 400 || item.statusCode === -1) {
        allGood = false;
        console.log(`Failed Request (${item.statusCode}): ${item.url}`);
      }
    });
  }
  if (allGood) console.log('All static resources loaded successfully (200 OK).');

} catch (e) {
  console.log('Failed to parse report:', e.message);
}
