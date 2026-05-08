# 本地 UI 检查与调试测试报告

## 1. 测试范围与环境
- 目标：验证现有网站在本地环境下的 UI/交互/响应式表现，定位并修复问题，输出可复现实验与指标
- 被测站点（生产构建预览）：`http://127.0.0.1:4174/Sparse2Full_Official/`
- 被测站点（开发模式）：`http://127.0.0.1:5173/Sparse2Full_Official/`

## 2. 自动化访问测试结果（Chromium）
已对页面可达性与核心交互进行自动化验证，结果见：
- [ui-audit.md](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/ui-audit.md)
- [ui-audit.json](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/ui-audit.json)
- 截图目录：[screenshots](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/screenshots)

已验证功能清单（Chromium Desktop/Mobile）：
- 页面可达性：Home / Docs / About / Contact / Login / Admin 均返回 200
- 导航与跳转：路由跳转正常，刷新后不会白屏（GitHub Pages SPA fallback 已补齐）
- 搜索：`Ctrl/Cmd + K` 唤起搜索；搜索结果可跳转到 `/docs#<section>`
- Contact 表单：必填校验、邮箱格式校验、成功提交提示
- Login 表单：必填校验；Mock 登录后跳转 Admin

## 3. 发现的问题与修复记录
### 3.1 搜索快捷键无效 + 搜索结果无法定位到 Docs
- 现象：按 `Ctrl/Cmd + K` 不能打开搜索；点击结果只修改 hash，非 Docs 页面无法定位
- 根因：
  - 快捷键监听在弹窗内部且逻辑恒等于 close（无法 open）
  - 结果链接为 `#id`，在非 `/docs` 页面无对应 section
- 修复：
  - 将 `Ctrl/Cmd + K` 切换逻辑移动到 Layout 全局监听，并支持 Escape 关闭
  - 搜索结果改为导航到 `/docs#id` 并在 Docs 页面监听 hash 进行滚动定位
- 代码：
  - [Layout.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/components/Layout.jsx#L1-L46)
  - [SearchModal.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/components/SearchModal.jsx#L1-L107)
  - [Docs.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/pages/Docs.jsx#L1-L235)

### 3.2 Contact 表单不可用（无提交逻辑）
- 现象：点击“发送消息”无行为；无错误提示与成功提示
- 修复：补齐 submit、校验、错误提示与提交成功提示（ARIA 支持）
- 代码：
  - [Contact.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/pages/Contact.jsx#L1-L134)

### 3.3 Login 表单使用 Link 伪提交、存在 `href="#"` 坏链
- 现象：登录按钮仅跳转，不做表单校验；注册/忘记密码为 `#`
- 修复：改为真实 `form onSubmit` 校验并 Mock 登录跳转；将 `#` 链接替换为可用页面（Contact）
- 代码：
  - [Login.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/pages/Login.jsx#L1-L118)

### 3.4 favicon 404 + CSP 导致 GA4 请求报错
- 现象：
  - `index.html` 引用 `/vite.svg`，public 中实际为 `favicon.svg`
  - CSP 未允许 GA endpoint，控制台出现连接失败/被拦截噪声
- 修复：favicon 指向 `/favicon.svg`；CSP 明确放行 GA 连接域名
- 代码：
  - [index.html](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/index.html#L1-L44)

### 3.5 GitHub Pages 刷新 404（SPA 深链问题）
- 现象：GitHub Pages 上直接刷新 `/docs` 等路径会 404（典型 SPA 问题）
- 修复：新增 `public/404.html` 做 fallback，将路径回写到 `/?p=...` 并在 `index.html` 解析恢复
- 代码：
  - [404.html](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/public/404.html)
  - [index.html](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/index.html#L1-L44)

### 3.6 性能问题：主包过大导致 Lighthouse 性能分偏低
- 现象：生产预览初始 Lighthouse Performance ≈ 79（TBT 偏高）
- 根因：`react-syntax-highlighter` 及 Docs 相关代码被打进主 bundle
- 修复：
  - 路由级代码分割（pages 全部改为 React.lazy 动态加载）
  - Docs 里高亮组件改为动态 import，仅进入 Docs 时加载
- 代码：
  - [App.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/App.jsx#L1-L33)
  - [Docs.jsx](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/src/pages/Docs.jsx#L1-L235)

## 4. 性能指标对比（Lighthouse）
### 4.1 优化前（生产预览，未分包）
- 报告文件：[lh-preview-desktop.json](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/lighthouse/lh-preview-desktop.json)
- Performance 79 / Accessibility 92 / Best Practices 96 / SEO 100
- FCP 2.6s / LCP 2.6s / TTI 3.5s / TBT 570ms / CLS 0

### 4.2 优化后（生产预览，路由分包 + Docs 懒加载）
- Desktop 报告文件：[lh-preview-desktop-split.json](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/lighthouse/lh-preview-desktop-split.json)
- Mobile 报告文件：[lh-preview-mobile.json](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/lighthouse/lh-preview-mobile.json)
- Performance 99 / Accessibility 92 / Best Practices 100 / SEO 100
- FCP 1.8s / LCP 1.9s / TTI 2.9s / TBT 10ms / CLS 0

## 5. 跨浏览器兼容性说明（本机环境限制）
Playwright 自动化尝试启动 Firefox/WebKit 失败（环境缺少系统依赖与 libstdc++ 版本不匹配），因此在当前运行环境中无法完成真实 Firefox/Safari(WebKit) 的本地自动化回归。
- 失败详情已记录在 [ui-audit.md](file:///share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/minimal_export/website/test-artifacts/ui-audit.md)
- 建议做法：
  - 使用官方 Playwright Docker 镜像运行 E2E
  - 或在具备依赖的 Ubuntu/Windows/macOS CI runner 上执行三浏览器矩阵

