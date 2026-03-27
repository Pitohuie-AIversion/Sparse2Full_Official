# Tasks
- [x] Task 1: 初始化前端项目基础架构
  - [x] SubTask 1.1: 在 `website` 目录下使用 Vite + React + Tailwind CSS 初始化项目
  - [x] SubTask 1.2: 安装和配置必要的依赖（如 `lucide-react` 图标库, `react-syntax-highlighter` 代码高亮, `fuse.js` 模糊搜索等）
- [x] Task 2: 页面整体布局与基础组件开发
  - [x] SubTask 2.1: 设计并实现具有专业科技感的主题和配色（推荐暗黑极客风/科技蓝风格）
  - [x] SubTask 2.2: 实现响应式顶部导航栏（包含 Logo、仓库链接、演示地址、状态徽章、搜索入口）
  - [x] SubTask 2.3: 实现左侧粘性侧边栏目录（Table of Contents），支持滚动自动高亮当前阅读章节
- [x] Task 3: 核心内容区块开发（一）- 项目概述与架构
  - [x] SubTask 3.1: 开发 Hero 区块（项目概述、主要功能展示、部署状态徽章、构建历史简览）
  - [x] SubTask 3.2: 开发“功能特性列表”与“技术栈介绍”展示区，使用卡片网格布局
  - [x] SubTask 3.3: 开发“项目结构展示”组件（树状图样式并带语法高亮）
- [x] Task 4: 核心内容区块开发（二）- 部署与配置文档
  - [x] SubTask 4.1: 开发“GitHub Pages 配置步骤”与“部署流程说明”区块，集成代码高亮功能显示 YAML 配置文件
  - [x] SubTask 4.2: 开发“分支管理策略”说明区块（包含 master/gh-pages 等分支流向图说明或文本描述）
- [x] Task 5: 核心内容区块开发（三）- 扩展文档与社区
  - [x] SubTask 5.1: 开发“使用示例”区块
  - [x] SubTask 5.2: 开发“常见问题解答(FAQ)”，使用手风琴(Accordion)组件展示
  - [x] SubTask 5.3: 开发“版本更新日志(Changelog)”、“相关文档链接”、“贡献指南”、“许可证(License)”与“联系方式”区块
- [x] Task 6: 搜索功能与体验优化
  - [x] SubTask 6.1: 集成全局搜索弹窗（基于 Fuse.js 的内存数据过滤），支持快捷键唤起
  - [x] SubTask 6.2: 检查并优化移动端适配效果（折叠导航、排版调整）
  - [x] SubTask 6.3: 补充必要的动画过渡（平滑滚动、模块渐显动画），提升交互体验和视觉品质

# Task Dependencies
- Task 2 depends on Task 1
- Task 3, Task 4, Task 5 depend on Task 2
- Task 6 depends on Task 3, 4, 5
