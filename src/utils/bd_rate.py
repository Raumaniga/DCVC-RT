"""
BD-Rate (Bjøntegaard Delta Rate) Calculation
=============================================
Thuật toán chuẩn VCEG để tính BD-Rate và BD-Metric (BD-PSNR / BD-Accuracy).

BD-Rate: Phần trăm tiết kiệm bitrate tại cùng mức chất lượng.
BD-Metric: Sự cải thiện chất lượng (dB hoặc %) tại cùng mức bitrate.

Tham khảo: G. Bjøntegaard, "Calculation of average PSNR differences between
           RD-curves," ITU-T VCEG-M33, April 2001.

Sử dụng:
    from src.utils.bd_rate import compute_bd_rate, compute_bd_metric

    # anchor và test là 2 bộ (bpp, metric) — ít nhất 4 điểm mỗi bộ
    bd_r = compute_bd_rate(anchor_bpp, anchor_metric, test_bpp, test_metric)
    bd_m = compute_bd_metric(anchor_bpp, anchor_metric, test_bpp, test_metric)
"""

import numpy as np
from scipy import interpolate


def _check_inputs(rate, metric):
    """Kiểm tra và sắp xếp inputs theo thứ tự rate tăng dần."""
    rate = np.array(rate, dtype=np.float64)
    metric = np.array(metric, dtype=np.float64)

    if len(rate) != len(metric):
        raise ValueError(
            f"Số lượng rate ({len(rate)}) và metric ({len(metric)}) phải bằng nhau."
        )
    if len(rate) < 4:
        raise ValueError(
            f"Cần ít nhất 4 điểm RD, chỉ có {len(rate)} điểm."
        )

    # Sắp xếp theo rate tăng dần
    idx = np.argsort(rate)
    rate = rate[idx]
    metric = metric[idx]

    # Kiểm tra không có rate trùng nhau
    if len(np.unique(rate)) != len(rate):
        raise ValueError("Các giá trị rate không được trùng nhau.")

    return rate, metric


def _pchip_integrate(x, y, x_min, x_max):
    """
    Tích phân hàm nội suy PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    trên khoảng [x_min, x_max].

    PCHIP được ưu tiên hơn cubic spline vì giữ được tính đơn điệu cục bộ,
    tránh dao động (Runge phenomenon) thường gặp với polynomial fitting.
    """
    pchip = interpolate.PchipInterpolator(x, y)
    result, _ = interpolate.splrep(x, y, k=3, s=0), None

    # Dùng quadrature cho tích phân chính xác
    from scipy.integrate import quad
    integral, _ = quad(pchip, x_min, x_max)
    return integral


def compute_bd_rate(anchor_rate, anchor_metric, test_rate, test_metric):
    """
    Tính BD-Rate (%) giữa anchor và test.

    BD-Rate < 0: test tốt hơn anchor (tiết kiệm bitrate)
    BD-Rate > 0: test tệ hơn anchor (tốn thêm bitrate)

    Args:
        anchor_rate: BPP của anchor (list/array, ≥4 điểm)
        anchor_metric: Metric (PSNR/accuracy) của anchor
        test_rate: BPP của test
        test_metric: Metric của test

    Returns:
        bd_rate: Phần trăm thay đổi bitrate (%)
    """
    anchor_rate, anchor_metric = _check_inputs(anchor_rate, anchor_metric)
    test_rate, test_metric = _check_inputs(test_rate, test_metric)

    # Chuyển sang log domain cho rate
    anchor_log_rate = np.log10(anchor_rate)
    test_log_rate = np.log10(test_rate)

    # Xác định khoảng metric chung để tính tích phân
    metric_min = max(anchor_metric.min(), test_metric.min())
    metric_max = min(anchor_metric.max(), test_metric.max())

    if metric_min >= metric_max:
        raise ValueError(
            f"Không có khoảng metric chồng lấp giữa anchor [{anchor_metric.min():.4f}, "
            f"{anchor_metric.max():.4f}] và test [{test_metric.min():.4f}, "
            f"{test_metric.max():.4f}]."
        )

    # Nội suy: metric → log_rate (đảo trục)
    anchor_interp = interpolate.PchipInterpolator(anchor_metric, anchor_log_rate)
    test_interp = interpolate.PchipInterpolator(test_metric, test_log_rate)

    # Tích phân trên khoảng metric chung
    from scipy.integrate import quad
    anchor_integral, _ = quad(anchor_interp, metric_min, metric_max)
    test_integral, _ = quad(test_interp, metric_min, metric_max)

    # BD-Rate = (10^(avg_diff) - 1) * 100%
    avg_diff = (test_integral - anchor_integral) / (metric_max - metric_min)
    bd_rate = (10 ** avg_diff - 1) * 100.0

    return bd_rate


def compute_bd_metric(anchor_rate, anchor_metric, test_rate, test_metric):
    """
    Tính BD-Metric (BD-PSNR hoặc BD-Accuracy).

    BD-Metric > 0: test tốt hơn anchor (cải thiện chất lượng)
    BD-Metric < 0: test tệ hơn anchor

    Args:
        anchor_rate: BPP của anchor (list/array, ≥4 điểm)
        anchor_metric: Metric (PSNR/accuracy) của anchor
        test_rate: BPP của test
        test_metric: Metric của test

    Returns:
        bd_metric: Trung bình thay đổi metric (dB hoặc %)
    """
    anchor_rate, anchor_metric = _check_inputs(anchor_rate, anchor_metric)
    test_rate, test_metric = _check_inputs(test_rate, test_metric)

    # Chuyển sang log domain cho rate
    anchor_log_rate = np.log10(anchor_rate)
    test_log_rate = np.log10(test_rate)

    # Xác định khoảng log_rate chung
    log_rate_min = max(anchor_log_rate.min(), test_log_rate.min())
    log_rate_max = min(anchor_log_rate.max(), test_log_rate.max())

    if log_rate_min >= log_rate_max:
        raise ValueError(
            f"Không có khoảng rate chồng lấp giữa anchor và test."
        )

    # Nội suy: log_rate → metric
    anchor_interp = interpolate.PchipInterpolator(anchor_log_rate, anchor_metric)
    test_interp = interpolate.PchipInterpolator(test_log_rate, test_metric)

    # Tích phân trên khoảng log_rate chung
    from scipy.integrate import quad
    anchor_integral, _ = quad(anchor_interp, log_rate_min, log_rate_max)
    test_integral, _ = quad(test_interp, log_rate_min, log_rate_max)

    # BD-Metric = trung bình hiệu
    bd_metric = (test_integral - anchor_integral) / (log_rate_max - log_rate_min)

    return bd_metric


def compute_bd_rate_from_json(anchor_json_path, test_json_path, metric_key='ave_all_frame_psnr'):
    """
    Tính BD-Rate từ 2 file JSON kết quả test (format giống test_video.py output).

    Args:
        anchor_json_path: Đường dẫn file JSON anchor
        test_json_path: Đường dẫn file JSON test
        metric_key: Key của metric trong JSON (mặc định: 'ave_all_frame_psnr')

    Returns:
        dict: BD-Rate và BD-Metric cho từng sequence và trung bình
    """
    import json

    with open(anchor_json_path, 'r') as f:
        anchor_data = json.load(f)
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    results = {}
    all_bd_rates = []

    for ds_name in anchor_data:
        if ds_name not in test_data:
            continue
        results[ds_name] = {}

        for seq in anchor_data[ds_name]:
            if seq not in test_data[ds_name]:
                continue

            # Thu thập các rate point
            anchor_bpp = []
            anchor_metric = []
            test_bpp = []
            test_metric = []

            for rate_idx in sorted(anchor_data[ds_name][seq].keys()):
                point = anchor_data[ds_name][seq][rate_idx]
                anchor_bpp.append(point['ave_all_frame_bpp'])
                anchor_metric.append(point[metric_key])

            for rate_idx in sorted(test_data[ds_name][seq].keys()):
                point = test_data[ds_name][seq][rate_idx]
                test_bpp.append(point['ave_all_frame_bpp'])
                test_metric.append(point[metric_key])

            try:
                bd_r = compute_bd_rate(anchor_bpp, anchor_metric, test_bpp, test_metric)
                bd_m = compute_bd_metric(anchor_bpp, anchor_metric, test_bpp, test_metric)
                results[ds_name][seq] = {
                    'bd_rate': bd_r,
                    'bd_metric': bd_m,
                }
                all_bd_rates.append(bd_r)
            except ValueError as e:
                results[ds_name][seq] = {
                    'bd_rate': None,
                    'bd_metric': None,
                    'error': str(e),
                }

    if all_bd_rates:
        results['_average'] = {
            'bd_rate': np.mean(all_bd_rates),
            'num_sequences': len(all_bd_rates),
        }

    return results
