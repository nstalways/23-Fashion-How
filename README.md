# 2023 Fashion-How Season 4: Sub-Task3 Repository
## 대회 정보
- **주최 기관**: ETRI
- **대회 기간**: 08.02 ~ 09.15
- **참여 과제**: Sub-Task3 - Continual Learning
<br>

## 대회 결과
- **종합 등수**: 종합 우수 (2등 혹은 3등)
- **Task3 등수**: 4등
<br>

## 코드 사용법
- **주의사항**: 현재 코드는 Sub-Task3 실채점 리더보드에서 0.6320286 score를 기록한 실험을 재현하기 위한 코드입니다.
- **실험 환경**
  - Google Cloud Platform
  - Tesla T4 GPU
  - CUDA 11.6

1. `git clone` 명령어로 본 repository를 복사
1. Anaconda 혹은 Miniconda를 설치하고, `conda create -n fashion_how python==3.10.12` 명령어로 가상환경을 생성. (권장, 필수 X)
1. `pip install -r requirements.txt` 명령어를 이용하여 필요한 라이브러리 설치
1. [FASHION-HOW LEADERBOARD 3](https://fashion-how.org/ETRI23/fascode_board/season4/leaderboard3/leader3outline.html) 페이지로 이동한 뒤, [신규)학습용 데이터셋 다운로드(링크)](https://drive.google.com/file/d/1E0dVb4W_QuWpuUQyGPLLo7Md41gUerPD/view?usp=sharing)와 [Validation Set 다운로드 (링크)](https://drive.google.com/file/d/1ZNPzt3t9D8r-ZkE8yB3o9He2kqqFp2l_/view)를 눌러 데이터를 다운로드
1. 다운받은 데이터셋을 아래 형식과 동일하게 구성
    ```
    23-Fashion-How
      >> .git
      >> .github
      >> after_competition
      >> data
        >> dialogue
        >  >> eval
        >  >  >> cl_eval_task*.dev (dev로 끝나는 파일 모두를 저장. 총 12개)
        >  >
        >  >> tr
        >  >  >> task*.ddata.txt (txt로 끝나는 파일 모두를 저장. 총 12개)
        >  >
        >  >> tt
        >  >  >> cl_eval_task*.tst (tst로 끝나는 파일 모두를 저장. 총 12개)
        >  
        >> img_feats
        >> img_jpg
        >> sstm_v0p5_deploy
        >
        >> mdata.txt.2023.0823
        >> mdata.wst.txt.2023.0823
      >> ...
    ```
1. `cd sub_task3_custom` 명령어를 통해 이동
1. 터미널 상에서 `bash run_example.sh` 명령어를 실행하여 실험을 진행.
1. 모든 데이터에 대해 학습 및 평가가 완료된 후 재현이 올바르게 되었는지 확인하고 싶다면, `nohup_logs/21_18_dec_mem_size.out` 파일의 최종 score와 비교해보면 됨. 
**Validation으로 사용한 data는 .tst가 아닌 .dev인 점을 주의** (업데이트 예정)
<br>

## 팀 둥굴레 멤버
- _김준태_ 님: Sub-Task1, 2 참여
- _이다현_ 님: Sub-Task3, 4 참여
- _이준하_ 님: Sub-Task1, 4 참여
- _박수영_ 님: Sub-Task3 참여
- _정호찬_ 님: Sub-Task3 참여
