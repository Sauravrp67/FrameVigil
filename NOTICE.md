# Attribution and third-party notices

FrameVigil contributions are copyright (c) 2026 Saurav Raj Paudel and licensed
under the GNU Affero General Public License, version 3 only (AGPL-3.0-only).
See [LICENSE](LICENSE). Upstream material retains its copyright and license.

## Adapted code

The constant acceleration Kalman tracking and empty-scene classifier adapt
[ALBIREO](https://github.com/amirtaherin/albireo), copyright (c) 2024 Amir Taherin,
under the MIT license. Attribution remains in:

- `src/tracking/scheduler.py`
- `src/tracking/kalman.py`
- `src/detectors/empty_scene.py`

FrameVigil modifies and extends these components for variant comparisons and
measurement. The upstream permission and copyright notice is preserved in
[licenses/ALBIREO-MIT.txt](licenses/ALBIREO-MIT.txt).

## External implementations and assets

- [Ultralytics](https://github.com/ultralytics/ultralytics) provides the YOLO
  implementation and weights. Its open-source licensing is AGPL-3.0; commercial
  licensing is available separately from Ultralytics. The license text is retained
  in [licenses/ULTRALYTICS-LICENSE](licenses/ULTRALYTICS-LICENSE).
- [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) supplies the detector architecture;
  this project loads the [PekingU checkpoint](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
  through [Transformers](https://github.com/huggingface/transformers).
- The downloaded empty-scene classifier checkpoint comes from ALBIREO.
- Evaluation uses [pycocotools](https://github.com/ppwwyyxx/cocoapi) and
  [TrackEval](https://github.com/JonathonLuiten/TrackEval).
- Dataset annotations and demo imagery originate from
  [BDD100K](https://www.bdd100k.com/). Dataset terms apply to these assets,
  including imagery reproduced in the README demo.

Dependencies, downloaded weights, datasets and NVIDIA runtime software remain
subject to their respective upstream terms; FrameVigil's license does not
relicense those materials. Refer to each upstream project for its license and
preferred research citation. Academic citation does not replace license notices.
