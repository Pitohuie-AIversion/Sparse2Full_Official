# 安全与工程化进阶加固报告 (Security & Engineering Enhancements)

作为网站工程师，针对之前的交付内容与对话总结，进行了如下的二次加固与漏洞排查：

## 1. 依赖安全与漏洞修复 (Dependency Security)
- 执行了全面的 `npm audit` 检查，发现了 `vite`, `lodash`, `postcss` 等底层依赖存在 6 个（3高3中）安全漏洞。
- **修复方案**：执行 `npm audit fix`，将所有包含漏洞的包升级到了安全版本，当前审计已做到 **0 漏洞**。
- **自动化防范**：在项目根目录新增了 `.github/dependabot.yml`，配置了 `npm` 和 `github-actions` 每周自动检查并提交安全更新 PR，确保依赖安全不过期。

## 2. Web 安全响应头与防篡改 (Security Headers & SRI)
由于是静态托管网站（GitHub Pages），我们在 `index.html` 注入了更为严苛的 `meta` 安全策略：
- **CSP (Content-Security-Policy)**：限制仅从可信源加载脚本与图片。
- **HSTS**：强制全站 HTTPS。
- **X-Content-Type-Options: nosniff**：防止浏览器 MIME 类型嗅探攻击。
- **X-Frame-Options: DENY**：禁止网站被嵌入 iframe，防范点击劫持（Clickjacking）。
- **Referrer-Policy: strict-origin-when-cross-origin**：保护用户隐私与路由跳转。
- **Permissions-Policy**：禁用地理位置、麦克风和摄像头权限，收敛浏览器 API 攻击面。
- **SRI (Subresource Integrity)**：在 Vite 中引入了 `@small-tech/vite-plugin-sri`，构建时自动为所有 `<script>` 和 `<link rel="stylesheet">` 生成 `integrity` 校验哈希，防止 CDN 或资源被篡改。

## 3. SEO 与爬虫友好化 (SEO & Crawlability)
- **元数据丰富**：在 `index.html` 补充了完整的 `Open Graph (OG)` 与 `Twitter Cards` 标签，提升了链接在社交媒体分享时的卡片展示效果（包含指定配图、描述）。
- **爬虫指引**：在 `public/` 目录下新增了 `robots.txt` 与 `sitemap.xml`。
  - `robots.txt` 允许全站抓取，并指明了 Sitemap 地址。
  - `sitemap.xml` 显式列出了各个子页面的路由、更新频率与抓取优先级。

## 4. CI/CD 安全与 SAST (SAST & Pipeline)
- 引入了 **GitHub CodeQL** 高级安全分析：
  - 新增 `.github/workflows/codeql.yml` 工作流。
  - 在每次代码合并到 `main` 分支时，以及每周定时运行静态应用安全测试 (SAST)，扫描 OWASP Top 10 漏洞与不安全代码实践。
- 确保 `deploy.yml` 遵循最小权限原则（`permissions` 设置明确）。

## 5. 前端监控与错误追踪 (Monitoring)
- 在原有的 GA4 (Google Analytics) PV/UV 埋点基础上，于 `src/main.jsx` 顶层增加了 `window.addEventListener('error')` 和 `unhandledrejection` 监听器。
- 该机制可捕获所有未被 React ErrorBoundary 捕获的 JS 异常和 Promise 拒绝，并留出了自研日志上报的 `fetch` 接口位。

以上加固已全部落地并随代码推送到 `main` 分支，目前项目在安全性、SEO 和工程化方面均达到了极高的企业级标准。