# 설정 (처음 한 번)
1. GitHub에서 새 저장소 생성 (Public 권장, 이름 자유)
2. 이 폴더의 파일 전체를 업로드 (.github/workflows/update.yml 경로 유지, docs 폴더 포함)
3. Settings → Pages → Source: "Deploy from a branch" → Branch: main, 폴더: /docs → Save
4. Actions 탭 → update-dashboard → Run workflow (첫 실행, 이후 매일 자동)
5. 1~2분 뒤 https://<아이디>.github.io/<저장소명>/ 를 즐겨찾기

# 수정
generate.py 맨 위 CONFIG에서 임계값·비중 규칙만 바꾸면 됩니다.
