# Model and software licenses

Licenses for the evaluated software and pretrained weights, checked on 2026-10-07. Links point to official releases; full versions and source records are available in [models.json](models.json) and [sources.json](sources.json).

| Model / configuration | Software version | Code license | Weight license |
| --- | --- | --- | --- |
| TabPFNv2 | `tabpfn==8.0.2` | [Prior Labs License v1.2*](https://pypi.org/project/tabpfn/8.0.2/) | [Prior Labs License v1.1*](https://github.com/PriorLabs/TabPFN/blob/49394b053a6759cfe68e90c21a2d51c31b396768/LICENSE) |
| TabPFNv2.5 | `tabpfn==8.0.2` | [Prior Labs License v1.2*](https://pypi.org/project/tabpfn/8.0.2/) | [TABPFN-2.5 License v1.1†](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE) |
| TabPFNv3 | `tabpfn==8.0.2` | [Prior Labs License v1.2*](https://pypi.org/project/tabpfn/8.0.2/) | [TABPFN-3 License v1.0†](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE) |
| TabPFNv3.5 | `9.1.0 / 15f5e6b2` | [Apache-2.0](https://github.com/PriorLabs/TabPFN/blob/15f5e6b2b629b905879b9be907261416f20d0df5/LICENSE) | [TABPFN-3.5 License v1.0†](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE) |
| TabICLv1 | `tabicl==2.1.0` | [BSD-3-Clause](https://pypi.org/project/tabicl/2.1.0/) | [BSD-3-Clause](https://huggingface.co/jingang/TabICL/blob/main/README.md) |
| TabICLv2 | `tabicl==2.1.0` | [BSD-3-Clause](https://pypi.org/project/tabicl/2.1.0/) | [BSD-3-Clause](https://huggingface.co/jingang/TabICL/blob/main/README.md) |
| TabDPT1.1 | `tabdpt==1.1.13` | [Apache-2.0](https://pypi.org/project/tabdpt/1.1.13/) | [Apache-2.0](https://huggingface.co/Layer6/TabDPT/blob/main/README.md) |
| TabDPT1.3 | `1.3.1 / 93670551` | [Apache-2.0](https://github.com/layer6ai-labs/TabDPT-inference/blob/93670551adba28b186354fab9a1584c56c34aa76/LICENSE) | [Apache-2.0](https://huggingface.co/Layer6/TabDPT/blob/main/README.md) |
| Causilo | `1.0.3 / 4d26d497` | [Apache-2.0](https://github.com/nums-ai/causilo/blob/4d26d497de28734db52c6bfc2ea949c12cc17308/LICENSE) | [Causilo License v1.0](https://huggingface.co/nums-ai/causilo/blob/94f2bd91db0737d4da59f347910662905ecb5a09/LICENSE) |
| LimiX-2 | `516bf396` | [Stable AI Technology Co., Ltd. License v1.0](https://github.com/limix-ldm-ai/LimiX/blob/516bf396333feb3198cf7aff8a6c10421f218e24/LICENSE.txt) | [Stable AI Technology Co., Ltd. License v1.0](https://huggingface.co/stable-ai/LimiX-2/blob/main/LICENSE) |
| TabFM | `1.0.1 / fbb66556` | [Apache-2.0](https://github.com/google-research/tabfm/blob/fbb665569425fd2f490c6576b3af967876fe11ff/LICENSE) | [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE) |
| RealMLP / RealMLP-HPO | `pytabkit==1.7.3` | [Apache-2.0](https://pypi.org/project/pytabkit/1.7.3/) | — |
| XGBoost / XGBoost-HPO | `xgboost==3.2.0` | [Apache-2.0](https://pypi.org/project/xgboost/3.2.0/) | — |
| BART | `bartz>=0.12,<0.13` | [MIT](https://pypi.org/project/bartz/0.12.0/) | — |

\* Prior Labs licenses add attribution provisions to Apache-2.0. † These TabPFN weight licenses permit non-commercial, non-production use. Causilo and TabFM weights have separate research/non-commercial terms. LimiX uses an Apache-derived license with additional attribution and model-naming provisions. See the linked texts for conditions.

Baselines are trained per split and use no pretrained weights. The BART license was checked for bartz 0.12.0 within the specified version range. TabICL entries cover the tabular code; its forecasting component has separate notices.

## Bundled code

- **LimiX configurations:** [license](../evaluation/adapters/configs/LIMIX_LICENSE.txt).
- **PCP:** [official source](https://github.com/yaozhang24/pcp/tree/7a4b33ee852b95bd65f8fe5486b4d7e6ded24bad), [provenance](../conformal/_vendor/pcp/SOURCE.json). No explicit license was found at this revision.

Original project code and documentation are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Third-party assets retain the licenses listed above.
