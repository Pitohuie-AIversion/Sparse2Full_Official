# UI 测试报告
- Base URL: http://localhost:4173/Sparse2Full_Official/
- Started: 2026-05-08T06:44:10.685Z
- Finished: 2026-05-08T06:44:29.490Z
- Passes: 12  Fails: 2

## chromium / desktop
### 页面可达性
- ✅ home: 200 (1152ms)
- ✅ docs: 200 (1009ms)
- ✅ about: 200 (797ms)
- ✅ contact: 200 (657ms)
- ✅ login: 200 (674ms)
- ✅ admin: 200 (669ms)

### 功能检查
- ✅ Search hotkey + navigate
- ✅ Contact form validation + submit
- ✅ Login validation + navigate to admin

## chromium / mobile
### 页面可达性
- ✅ home: 200 (1371ms)
- ✅ docs: 200 (1004ms)
- ✅ about: 200 (811ms)
- ✅ contact: 200 (655ms)
- ✅ login: 200 (674ms)
- ✅ admin: 200 (656ms)

### 功能检查
- ✅ Search hotkey + navigate
- ✅ Contact form validation + submit
- ✅ Login validation + navigate to admin

## firefox / n/a
- ❌ Browser launch failed: browserType.launch: Failed to launch the browser process.
Browser logs:

<launching> /share/fandixiaLab/suguangsheng/.cache/ms-playwright/firefox-1511/firefox/firefox -no-remote -headless -profile /tmp/playwright_firefoxdev_profile-2MANxt -juggler-pipe -silent
<launched> pid=3423848
[pid=3423848][err] XPCOMGlueLoad error for file /share/fandixiaLab/suguangsheng/.cache/ms-playwright/firefox-1511/firefox/libmozsandbox.so:
[pid=3423848][err] /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /share/fandixiaLab/suguangsheng/.cache/ms-playwright/firefox-1511/firefox/libmozsandbox.so)
[pid=3423848][err] Couldn't load XPCOM.
[pid=3423848] <process did exit: exitCode=255, signal=null>
[pid=3423848] starting temporary directories cleanup
Call log:
[2m  - <launching> /share/fandixiaLab/suguangsheng/.cache/ms-playwright/firefox-1511/firefox/firefox -no-remote -headless -profile /tmp/playwright_firefoxdev_profile-2MANxt -juggler-pipe -silent[22m
[2m  - <launched> pid=3423848[22m
[2m  - [pid=3423848][err] XPCOMGlueLoad error for file /share/fandixiaLab/suguangsheng/.cache/ms-playwright/firefox-1511/firefox/libmozsandbox.so:[22m
[2m  - [pid=3423848][err] /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /share/fandixiaLab/suguangsheng/.cache/ms-playwright/firefox-1511/firefox/libmozsandbox.so)[22m
[2m  - [pid=3423848][err] Couldn't load XPCOM.[22m
[2m  - [pid=3423848] <process did exit: exitCode=255, signal=null>[22m
[2m  - [pid=3423848] starting temporary directories cleanup[22m
[2m  - [pid=3423848] <gracefully close start>[22m
[2m  - [pid=3423848] <kill>[22m
[2m  - [pid=3423848] <skipped force kill spawnedProcess.killed=false processClosed=true>[22m
[2m  - [pid=3423848] finished temporary directories cleanup[22m
[2m  - [pid=3423848] <gracefully close end>[22m


## webkit / n/a
- ❌ Browser launch failed: browserType.launch: 
╔══════════════════════════════════════════════════════╗
║ Host system is missing dependencies to run browsers. ║
║ Missing libraries:                                   ║
║     libgtk-4.so.1                                    ║
║     libvulkan.so.1                                   ║
║     libgraphene-1.0.so.0                             ║
║     libicudata.so.74                                 ║
║     libicui18n.so.74                                 ║
║     libatomic.so.1                                   ║
║     libicuuc.so.74                                   ║
║     libevent-2.1.so.7                                ║
║     libflite.so.1                                    ║
║     libflite_usenglish.so.1                          ║
║     libflite_cmu_grapheme_lang.so.1                  ║
║     libflite_cmu_grapheme_lex.so.1                   ║
║     libflite_cmu_indic_lang.so.1                     ║
║     libflite_cmu_indic_lex.so.1                      ║
║     libflite_cmulex.so.1                             ║
║     libflite_cmu_time_awb.so.1                       ║
║     libflite_cmu_us_awb.so.1                         ║
║     libflite_cmu_us_kal16.so.1                       ║
║     libflite_cmu_us_kal.so.1                         ║
║     libflite_cmu_us_rms.so.1                         ║
║     libflite_cmu_us_slt.so.1                         ║
║     libavif.so.16                                    ║
║     libjpeg.so.8                                     ║
║     libmanette-0.2.so.0                              ║
║     libx264.so                                       ║
╚══════════════════════════════════════════════════════╝
