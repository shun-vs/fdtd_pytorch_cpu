# test_viewer.py
import trimesh

try:
    # シンプルな箱を作成
    box = trimesh.creation.box()
    
    # シーンに追加
    scene = trimesh.Scene([box])
    
    print("3Dビューアの起動を試みます...")
    # ビューアを表示（このウィンドウを手動で閉じるまで、プログラムはここで待機します）
    scene.show()
    print("ビューアが正常に閉じられました。")

except Exception as e:
    print("\nエラーが発生しました:")
    print(e)