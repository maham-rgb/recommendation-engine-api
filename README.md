# Recommendation Engine API

## Overview
This project exposes the recommendation engine through FastAPI APIs.

The recommendation pipeline is based on:
- Co-view graph candidate generation
- LightGBM ranking model
- Graph-priority blended scoring

---

## Run API

```bash
uvicorn app:app --reload
## API Testing Plan

### 1. Existing User Test
Test with a valid user and valid source post.

```json
{
  "user_id": 5299302,
  "source_post_id": 395744096,
  "top_k": 10
}
## API Test Results

- Existing user test: Passed, returned 200 with 10 recommendations.
- Cold-start user test: Passed, returned 200 with fallback recommendations.
- Invalid source post test: Passed, returned 200 with empty recommendation list.
- Large top_k test: Passed, returned 200 with 50 recommendations.

{
  "user_id": 5299302,
  "source_post_id": 395744096,
  "top_k": 50,
  "count": 50,
  "recommendations": [
    {
      "post_id": 18506037168,
      "final_score": 0.87948458455538,
      "phase2_score": 0.8682392646802972,
      "phase1_score": 0.9057236642639068,
      "in_graph": 1
    },
    {
      "post_id": 10440894517,
      "final_score": 0.7856603777211835,
      "phase2_score": 0.9927308937733095,
      "phase1_score": 0.302495840266223,
      "in_graph": 0
    },
    {
      "post_id": 10392763126,
      "final_score": 0.7777460532988159,
      "phase2_score": 0.9930381257808861,
      "phase1_score": 0.27539788417398553,
      "in_graph": 0
    },
    {
      "post_id": 16968743816,
      "final_score": 0.764286951310989,
      "phase2_score": 0.8201214025936694,
      "phase1_score": 0.6340065649847347,
      "in_graph": 1
    },
    {
      "post_id": 10461997041,
      "final_score": 0.7274195191159044,
      "phase2_score": 0.7684448197178562,
      "phase1_score": 0.6316938177113505,
      "in_graph": 0
    },
    {
      "post_id": 10022773764,
      "final_score": 0.7197624880427893,
      "phase2_score": 0.7830245760123833,
      "phase1_score": 0.5721509494470699,
      "in_graph": 0
    },
    {
      "post_id": 10145168466,
      "final_score": 0.7050459374599308,
      "phase2_score": 0.7713872317884917,
      "phase1_score": 0.5502495840266224,
      "in_graph": 0
    },
    {
      "post_id": 10613807877,
      "final_score": 0.7033730733365348,
      "phase2_score": 0.759375804349776,
      "phase1_score": 0.5727000343056389,
      "in_graph": 0
    },
    {
      "post_id": 10004505061,
      "final_score": 0.6982164637816499,
      "phase2_score": 0.7571484687455023,
      "phase1_score": 0.560708452199328,
      "in_graph": 0
    },
    {
      "post_id": 10002523774,
      "final_score": 0.6925495582237725,
      "phase2_score": 0.7440419974023784,
      "phase1_score": 0.5724005334736921,
      "in_graph": 0
    },
    {
      "post_id": 10632746294,
      "final_score": 0.6138943695848158,
      "phase2_score": 0.7904647400058815,
      "phase1_score": 0.20189683860232946,
      "in_graph": 0
    },
    {
      "post_id": 10532331945,
      "final_score": 0.6124195892241282,
      "phase2_score": 0.795818345116103,
      "phase1_score": 0.1844891588095201,
      "in_graph": 0
    },
    {
      "post_id": 10449817835,
      "final_score": 0.6115540798708218,
      "phase2_score": 0.7956515490111895,
      "phase1_score": 0.18199331854329714,
      "in_graph": 0
    },
    {
      "post_id": 10515756954,
      "final_score": 0.611329469784263,
      "phase2_score": 0.8080592686170736,
      "phase1_score": 0.15229327250770483,
      "in_graph": 0
    },
    {
      "post_id": 10529906433,
      "final_score": 0.6102261141976859,
      "phase2_score": 0.8070178692647256,
      "phase1_score": 0.15104535237459335,
      "in_graph": 0
    },
    {
      "post_id": 10361134268,
      "final_score": 0.6067780534034114,
      "phase2_score": 0.788379402809826,
      "phase1_score": 0.1830415714551108,
      "in_graph": 0
    },
    {
      "post_id": 10368944900,
      "final_score": 0.6025719192266832,
      "phase2_score": 0.7806805992913719,
      "phase1_score": 0.1869849990757431,
      "in_graph": 0
    },
    {
      "post_id": 10026145418,
      "final_score": 0.5989832347339437,
      "phase2_score": 0.7779285208978932,
      "phase1_score": 0.1814442336847281,
      "in_graph": 0
    },
    {
      "post_id": 10116824680,
      "final_score": 0.5983595192170094,
      "phase2_score": 0.7900655907209021,
      "phase1_score": 0.15104535237459335,
      "in_graph": 0
    },
    {
      "post_id": 10455787411,
      "final_score": 0.5951675146474209,
      "phase2_score": 0.8026002908760664,
      "phase1_score": 0.11115770344724814,
      "in_graph": 0
    },
    {
      "post_id": 10168199697,
      "final_score": 0.5934171127996724,
      "phase2_score": 0.7771461655692126,
      "phase1_score": 0.16471598967074558,
      "in_graph": 0
    },
    {
      "post_id": 10084891111,
      "final_score": 0.5919484839355098,
      "phase2_score": 0.7874244315951511,
      "phase1_score": 0.1358379393963471,
      "in_graph": 0
    },
    {
      "post_id": 10314857815,
      "final_score": 0.5918730519353169,
      "phase2_score": 0.7741274335048492,
      "phase1_score": 0.16661282827307503,
      "in_graph": 0
    },
    {
      "post_id": 881151932,
      "final_score": 0.5911039530320135,
      "phase2_score": 0.7965384878552443,
      "phase1_score": 0.11175670511114165,
      "in_graph": 0
    },
    {
      "post_id": 10585039387,
      "final_score": 0.5909065172071596,
      "phase2_score": 0.7858289431146698,
      "phase1_score": 0.1360875234229694,
      "in_graph": 0
    },
    {
      "post_id": 10153436796,
      "final_score": 0.5907264278463304,
      "phase2_score": 0.7722968628429028,
      "phase1_score": 0.16706207952099517,
      "in_graph": 0
    },
    {
      "post_id": 720424180,
      "final_score": 0.5906620747419608,
      "phase2_score": 0.7717128927556051,
      "phase1_score": 0.16821016604345773,
      "in_graph": 0
    },
    {
      "post_id": 10593133295,
      "final_score": 0.5899497164716639,
      "phase2_score": 0.7782655159646562,
      "phase1_score": 0.15054618432134875,
      "in_graph": 0
    },
    {
      "post_id": 10564400555,
      "final_score": 0.5877133677517602,
      "phase2_score": 0.7919942954296862,
      "phase1_score": 0.11105786983659922,
      "in_graph": 0
    },
    {
      "post_id": 10350964584,
      "final_score": 0.5853977656569096,
      "phase2_score": 0.7569376321046974,
      "phase1_score": 0.1851380772787381,
      "in_graph": 0
    },
    {
      "post_id": 10402395575,
      "final_score": 0.5851644764370202,
      "phase2_score": 0.7563476467917581,
      "phase1_score": 0.18573707894263158,
      "in_graph": 0
    },
    {
      "post_id": 10106355287,
      "final_score": 0.5840392759915675,
      "phase2_score": 0.7755051685233437,
      "phase1_score": 0.13728552675075642,
      "in_graph": 0
    },
    {
      "post_id": 10237341208,
      "final_score": 0.5829055398210995,
      "phase2_score": 0.7515161257407282,
      "phase1_score": 0.18948083934196605,
      "in_graph": 0
    },
    {
      "post_id": 10600931731,
      "final_score": 0.582136384379605,
      "phase2_score": 0.7521287655782889,
      "phase1_score": 0.1854874949160093,
      "in_graph": 0
    },
    {
      "post_id": 10651568516,
      "final_score": 0.581954080663843,
      "phase2_score": 0.7820980949546726,
      "phase1_score": 0.11495138065190705,
      "in_graph": 0
    },
    {
      "post_id": 10567534296,
      "final_score": 0.581752783404861,
      "phase2_score": 0.7757299896387327,
      "phase1_score": 0.12913930219249414,
      "in_graph": 0
    },
    {
      "post_id": 10023830201,
      "final_score": 0.580470161063257,
      "phase2_score": 0.7706203620771478,
      "phase1_score": 0.13678635869751185,
      "in_graph": 0
    },
    {
      "post_id": 10011923908,
      "final_score": 0.5796434617428724,
      "phase2_score": 0.7413423354111394,
      "phase1_score": 0.2023460898502496,
      "in_graph": 0
    },
    {
      "post_id": 10648694189,
      "final_score": 0.5793964624027088,
      "phase2_score": 0.7407113712955266,
      "phase1_score": 0.20299500831946757,
      "in_graph": 0
    },
    {
      "post_id": 10452247754,
      "final_score": 0.5790403623898307,
      "phase2_score": 0.7407160869890379,
      "phase1_score": 0.20179700499168055,
      "in_graph": 0
    },
    {
      "post_id": 10276397300,
      "final_score": 0.5765948287156754,
      "phase2_score": 0.7527720444984205,
      "phase1_score": 0.16551465855593692,
      "in_graph": 0
    },
    {
      "post_id": 10421652370,
      "final_score": 0.5741882074879059,
      "phase2_score": 0.7330998637989814,
      "phase1_score": 0.20339434276206325,
      "in_graph": 0
    },
    {
      "post_id": 10108072198,
      "final_score": 0.5729659102386928,
      "phase2_score": 0.796858835401603,
      "phase1_score": 0.05054908485856906,
      "in_graph": 0
    },
    {
      "post_id": 10012709330,
      "final_score": 0.570611952980667,
      "phase2_score": 0.7564516360071774,
      "phase1_score": 0.13698602591880968,
      "in_graph": 0
    },
    {
      "post_id": 10592329405,
      "final_score": 0.5700735480771613,
      "phase2_score": 0.7323931710506714,
      "phase1_score": 0.19132776113897104,
      "in_graph": 0
    },
    {
      "post_id": 10126471858,
      "final_score": 0.5685386937836826,
      "phase2_score": 0.7688275611219236,
      "phase1_score": 0.10119800332778703,
      "in_graph": 0
    },
    {
      "post_id": 10220017793,
      "final_score": 0.5650925108288956,
      "phase2_score": 0.7205338697603191,
      "phase1_score": 0.20239600665557406,
      "in_graph": 0
    },
    {
      "post_id": 10642257542,
      "final_score": 0.5650645206530277,
      "phase2_score": 0.766942189792706,
      "phase1_score": 0.09401662599377836,
      "in_graph": 0
    },
    {
      "post_id": 10116745127,
      "final_score": 0.564222171072884,
      "phase2_score": 0.7765705599672512,
      "phase1_score": 0.06874259698602714,
      "in_graph": 0
    },
    {
      "post_id": 10231583947,
      "final_score": 0.562795304998166,
      "phase2_score": 0.7608366491654334,
      "phase1_score": 0.10069883527454243,
      "in_graph": 0
    }
  ]
}