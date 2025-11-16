import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from PIL import Image


def load_and_downsample_image(image_path, target_dpi=50):
    """
    TIF画像を読み込み、指定したDPIにダウンサンプリング
    """
    # PIL で画像を開いてDPI情報を取得
    with Image.open(image_path) as img:
        # 元のDPIを取得（存在しない場合は72と仮定）
        original_dpi = img.info.get('dpi', (72, 72))[0]

        # リサイズ比率を計算
        scale = target_dpi / original_dpi

        # 新しいサイズを計算
        new_size = (int(img.width * scale), int(img.height * scale))

        # リサイズ
        img_resized = img.resize(new_size, Image.LANCZOS)

        # numpy配列に変換
        img_array = np.array(img_resized)

    print(f"元の画像サイズ: {img.size}, DPI: {original_dpi}")
    print(f"リサイズ後: {new_size}, DPI: {target_dpi}")
    print(f"総ピクセル数: {new_size[0] * new_size[1]}")

    return img_array


def rgb_to_lab_pixels(image_array):
    """
    RGB画像の全ピクセルをLab色空間に変換
    """
    # RGB値を0-1の範囲に正規化
    if image_array.max() > 1:
        image_array = image_array / 255.0

    # RGB to Lab変換
    lab_image = rgb2lab(image_array)

    # 画像を(height * width, 3)の形にリシェイプ
    height, width = lab_image.shape[:2]
    lab_pixels = lab_image.reshape(-1, 3)
    rgb_pixels = image_array.reshape(-1, 3)

    return lab_pixels, rgb_pixels


def lab_to_display_color(lab_values):
    """
    Lab値を表示用のRGB色に変換
    """
    # Lab値をrgb2labが期待する形式に変換
    # L: 0-100, a: -128~127, b: -128~127
    lab_normalized = lab_values.copy()

    # 単一ピクセルとして扱うために(1, 1, 3)にリシェイプ
    lab_reshaped = lab_normalized.reshape(-1, 1, 3)

    # Lab to RGB変換
    rgb_colors = lab2rgb(lab_reshaped)

    # (n, 3)にリシェイプ
    rgb_colors = rgb_colors.reshape(-1, 3)

    # 0-1の範囲にクリップ
    rgb_colors = np.clip(rgb_colors, 0, 1)

    return rgb_colors


def plot_lab_3d(lab_pixels, image_path, plot_size=3):
    """
    Lab値を3次元空間にプロット
    """
    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]

    # Lab値を表示用RGB色に変換
    colors = lab_to_display_color(lab_pixels)

    # 3Dプロット
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(a, b, L, c=colors, s=plot_size, alpha=0.6)

    ax.set_xlabel('a*', fontsize=12)
    ax.set_ylabel('b*', fontsize=12)
    ax.set_zlabel('L*', fontsize=12)
    ax.set_title(f'Lab Color Space - 3D\n{image_path}', fontsize=14)

    plt.tight_layout()


def plot_ab_planes(lab_pixels, L_values=[0, 20, 50], image_path='', plot_size=3):
    """
    指定したL値でのab平面をプロット
    """
    fig, axes = plt.subplots(1, len(L_values), figsize=(6 * len(L_values), 5))

    if len(L_values) == 1:
        axes = [axes]

    for idx, L_target in enumerate(L_values):
        # L値が近いピクセルを抽出（±5の範囲）
        tolerance = 5
        mask = np.abs(lab_pixels[:, 0] - L_target) < tolerance
        filtered_pixels = lab_pixels[mask]

        if len(filtered_pixels) == 0:
            print(f"警告: L={L_target}付近にピクセルがありません")
            axes[idx].text(0.5, 0.5, f'No pixels at L≈{L_target}',
                           ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-128, 127)
            axes[idx].set_ylim(-128, 127)
            continue

        a = filtered_pixels[:, 1]
        b = filtered_pixels[:, 2]

        # 表示用の色を生成
        colors = lab_to_display_color(filtered_pixels)

        axes[idx].scatter(a, b, c=colors, s=plot_size, alpha=0.6)
        axes[idx].set_xlabel('a*', fontsize=12)
        axes[idx].set_ylabel('b*', fontsize=12)
        axes[idx].set_title(f'L* ≈ {L_target} (±{tolerance})\n{len(filtered_pixels)} pixels', fontsize=13)
        axes[idx].set_xlim(-128, 127)
        axes[idx].set_ylim(-128, 127)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect('equal')

    fig.suptitle(f'a*b* Planes at Different L* values\n{image_path}', fontsize=14, fontweight='bold')
    plt.tight_layout()


def analyze_tif_image(image_path, target_dpi=50, L_values=[0, 20, 50], plot_size=3):
    """
    TIF画像を読み込み、Lab色空間に変換して可視化

    Parameters:
    -----------
    image_path : str
        TIF画像のファイルパス
    target_dpi : int
        ダウンサンプリング後のDPI（デフォルト: 50）
    L_values : list
        ab平面を表示するL値のリスト（デフォルト: [0, 20, 50]）
    plot_size : int
        プロットのサイズ（デフォルト: 3）
    """
    print(f"画像を読み込み中: {image_path}")

    # 画像を読み込んでダウンサンプリング
    image_array = load_and_downsample_image(image_path, target_dpi)

    print("RGB to Lab変換中...")
    # Lab色空間に変換
    lab_pixels, rgb_pixels = rgb_to_lab_pixels(image_array)

    print(f"Lab値の範囲:")
    print(f"  L*: {lab_pixels[:, 0].min():.2f} ~ {lab_pixels[:, 0].max():.2f}")
    print(f"  a*: {lab_pixels[:, 1].min():.2f} ~ {lab_pixels[:, 1].max():.2f}")
    print(f"  b*: {lab_pixels[:, 2].min():.2f} ~ {lab_pixels[:, 2].max():.2f}")

    # 3Dプロット
    print("3Dグラフを作成中...")
    plot_lab_3d(lab_pixels, image_path, plot_size)

    # ab平面プロット
    print("ab平面グラフを作成中...")
    plot_ab_planes(lab_pixels, L_values, image_path, plot_size)

    print("完了！")
    plt.show()


# 使用例
if __name__ == "__main__":
    # パラメータ設定
    IMAGE_PATH = './tif/input.tif'  # TIF画像のパス
    TARGET_DPI = 50  # ダウンサンプリング後のDPI
    L_VALUES = [0, 20, 50]  # ab平面を表示するL値
    PLOT_SIZE = 3  # プロットサイズ

    # 実行
    analyze_tif_image(
        image_path=IMAGE_PATH,
        target_dpi=TARGET_DPI,
        L_values=L_VALUES,
        plot_size=PLOT_SIZE
    )