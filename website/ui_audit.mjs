import fs from 'node:fs';
import path from 'node:path';
import { chromium, firefox, webkit } from 'playwright';

const BASE_URL = process.env.BASE_URL || 'http://localhost:4173/Sparse2Full_Official/';

const ARTIFACTS_DIR = path.resolve('./test-artifacts');
const SHOTS_DIR = path.join(ARTIFACTS_DIR, 'screenshots');
fs.mkdirSync(SHOTS_DIR, { recursive: true });

const nowIso = () => new Date().toISOString();

const viewports = [
  { name: 'desktop', viewport: { width: 1365, height: 768 }, userAgent: undefined, isMobile: false, deviceScaleFactor: 1 },
  { name: 'mobile', viewport: { width: 390, height: 844 }, userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1', isMobile: true, deviceScaleFactor: 2 },
];

const browsers = [
  { name: 'chromium', launcher: chromium },
  { name: 'firefox', launcher: firefox },
  { name: 'webkit', launcher: webkit },
];

const routes = [
  { name: 'home', path: '' },
  { name: 'docs', path: 'docs' },
  { name: 'about', path: 'about' },
  { name: 'contact', path: 'contact' },
  { name: 'login', path: 'login' },
  { name: 'admin', path: 'admin' },
];

function joinUrl(base, p) {
  if (!p) return base;
  return new URL(p.replace(/^\//, ''), base).toString();
}

function short(s, n = 250) {
  if (!s) return s;
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

async function run() {
  const report = {
    startedAt: nowIso(),
    baseUrl: BASE_URL,
    runs: [],
    summary: { passes: 0, fails: 0, warnings: 0 },
  };

  for (const b of browsers) {
    let browser;
    try {
      browser = await b.launcher.launch({ headless: true });
    } catch (e) {
      report.runs.push({
        browser: b.name,
        error: `Browser launch failed: ${e.message}`,
      });
      report.summary.fails += 1;
      continue;
    }

    for (const vp of viewports) {
      const context = await browser.newContext({
        viewport: vp.viewport,
        userAgent: vp.userAgent,
        isMobile: vp.isMobile,
        deviceScaleFactor: vp.deviceScaleFactor,
        locale: 'zh-CN',
      });
      const page = await context.newPage();

      const consoleMessages = [];
      const pageErrors = [];
      const failedRequests = [];
      const badResponses = [];
      const navTimings = [];

      page.on('console', (msg) => {
        if (msg.type() === 'error' || msg.type() === 'warning') {
          consoleMessages.push({ type: msg.type(), text: short(msg.text()) });
        }
      });
      page.on('pageerror', (err) => pageErrors.push(short(err.message)));
      page.on('requestfailed', (req) => failedRequests.push({ url: req.url(), failure: req.failure()?.errorText || 'unknown' }));
      page.on('response', (res) => {
        const status = res.status();
        if (status >= 400) badResponses.push({ status, url: res.url() });
      });

      const runEntry = {
        browser: b.name,
        viewport: vp.name,
        pages: [],
        checks: [],
        consoleMessages,
        pageErrors,
        failedRequests,
        badResponses,
      };

      for (const r of routes) {
        const url = joinUrl(BASE_URL, r.path);
        const start = Date.now();
        let mainStatus = null;
        try {
          const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20000 });
          await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
          mainStatus = resp?.status() ?? null;
        } catch (e) {
          runEntry.pages.push({ name: r.name, url, ok: false, error: e.message, mainStatus });
          report.summary.fails += 1;
          continue;
        }
        const elapsedMs = Date.now() - start;
        navTimings.push({ name: r.name, elapsedMs, mainStatus });
        await page.waitForTimeout(50);
        const shotPath = path.join(SHOTS_DIR, `${b.name}_${vp.name}_${r.name}.png`);
        await page.screenshot({ path: shotPath, fullPage: true });
        runEntry.pages.push({ name: r.name, url, ok: mainStatus && mainStatus < 400, mainStatus, elapsedMs, screenshot: path.relative(ARTIFACTS_DIR, shotPath) });
        if (!mainStatus || mainStatus >= 400) report.summary.fails += 1;
        else report.summary.passes += 1;
      }

      try {
        await page.goto(joinUrl(BASE_URL, ''), { waitUntil: 'networkidle', timeout: 15000 });
        await page.keyboard.down(process.platform === 'darwin' ? 'Meta' : 'Control');
        await page.keyboard.press('KeyK');
        await page.keyboard.up(process.platform === 'darwin' ? 'Meta' : 'Control');
        await page.getByPlaceholder('搜索文档内容...').fill('部署');
        await page.getByRole('button', { name: /部署指南/ }).first().click();
        runEntry.checks.push({ name: 'Search hotkey + navigate', ok: page.url().includes('/docs#') });
      } catch (e) {
        runEntry.checks.push({ name: 'Search hotkey + navigate', ok: false, error: e.message });
        report.summary.fails += 1;
      }

      try {
        await page.goto(joinUrl(BASE_URL, 'contact'), { waitUntil: 'networkidle', timeout: 15000 });
        await page.getByLabel('姓名').fill('测试用户');
        await page.getByLabel('邮箱').fill('not-an-email');
        await page.getByLabel('留言').fill('hello');
        await page.getByRole('button', { name: '发送消息' }).click();
        await page.getByText('邮箱格式不正确').waitFor({ state: 'visible', timeout: 3000 });
        const emailErrorVisible = await page.getByText('邮箱格式不正确').isVisible();
        await page.getByLabel('邮箱').fill('tester@example.com');
        await page.getByRole('button', { name: '发送消息' }).click();
        await page.getByText('已收到消息').waitFor({ state: 'visible', timeout: 5000 });
        const successVisible = await page.getByText('已收到消息').isVisible();
        runEntry.checks.push({ name: 'Contact form validation + submit', ok: emailErrorVisible && successVisible });
      } catch (e) {
        runEntry.checks.push({ name: 'Contact form validation + submit', ok: false, error: e.message });
        report.summary.fails += 1;
      }

      try {
        await page.goto(joinUrl(BASE_URL, 'login'), { waitUntil: 'networkidle', timeout: 15000 });
        await page.getByRole('button', { name: /登录/ }).click();
        const err1 = await page.getByText('请输入邮箱地址').isVisible();
        await page.getByLabel('邮箱地址').fill('tester@example.com');
        await page.getByRole('button', { name: /登录/ }).click();
        const err2 = await page.getByText('请输入密码').isVisible();
        await page.getByLabel('密码').fill('password');
        await page.getByRole('button', { name: /登录/ }).click();
        await page.waitForURL('**/admin', { timeout: 8000 });
        runEntry.checks.push({ name: 'Login validation + navigate to admin', ok: err1 && err2 && page.url().includes('/admin') });
      } catch (e) {
        runEntry.checks.push({ name: 'Login validation + navigate to admin', ok: false, error: e.message });
        report.summary.fails += 1;
      }

      runEntry.timings = navTimings;
      report.runs.push(runEntry);

      await context.close();
    }

    await browser.close();
  }

  report.finishedAt = nowIso();

  const jsonPath = path.join(ARTIFACTS_DIR, 'ui-audit.json');
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));

  const mdLines = [];
  mdLines.push(`# UI 测试报告`);
  mdLines.push(`- Base URL: ${BASE_URL}`);
  mdLines.push(`- Started: ${report.startedAt}`);
  mdLines.push(`- Finished: ${report.finishedAt}`);
  mdLines.push(`- Passes: ${report.summary.passes}  Fails: ${report.summary.fails}`);
  mdLines.push('');

  for (const runEntry of report.runs) {
    mdLines.push(`## ${runEntry.browser || 'unknown'} / ${runEntry.viewport || 'n/a'}`);
    if (runEntry.error) {
      mdLines.push(`- ❌ ${runEntry.error}`);
      mdLines.push('');
      continue;
    }
    mdLines.push(`### 页面可达性`);
    for (const p of runEntry.pages || []) {
      mdLines.push(`- ${p.ok ? '✅' : '❌'} ${p.name}: ${p.mainStatus ?? 'n/a'} (${p.elapsedMs ?? 'n/a'}ms)${p.error ? ` — ${short(p.error, 120)}` : ''}`);
    }
    mdLines.push('');

    mdLines.push(`### 功能检查`);
    for (const c of runEntry.checks || []) {
      mdLines.push(`- ${c.ok ? '✅' : '❌'} ${c.name}${c.error ? ` — ${short(c.error, 120)}` : ''}`);
    }
    mdLines.push('');

    if ((runEntry.badResponses || []).length > 0) {
      mdLines.push(`### 非 200/300 资源`);
      for (const br of runEntry.badResponses.slice(0, 50)) {
        mdLines.push(`- ${br.status}: ${br.url}`);
      }
      mdLines.push('');
    }
    if ((runEntry.pageErrors || []).length > 0 || (runEntry.consoleMessages || []).length > 0) {
      mdLines.push(`### Console / JS 错误`);
      for (const e of runEntry.pageErrors || []) mdLines.push(`- ❌ pageerror: ${e}`);
      for (const m of runEntry.consoleMessages || []) mdLines.push(`- ⚠️ console ${m.type}: ${m.text}`);
      mdLines.push('');
    }
  }

  const mdPath = path.join(ARTIFACTS_DIR, 'ui-audit.md');
  fs.writeFileSync(mdPath, mdLines.join('\n'));

  console.log(`Wrote: ${jsonPath}`);
  console.log(`Wrote: ${mdPath}`);
}

await run();
