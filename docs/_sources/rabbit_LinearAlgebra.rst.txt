
線形代数
========

目標：線形代数を基礎から学び、特異値分解できるようになる
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # ライブラリ読み込み
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # 線形代数用
    import scipy.linalg as linalg

◼︎ベクトル(Vector)・・・いくつかの要素を持つひとまとまり
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一郎、二郎、三郎くんの身長をまとめたものは身長ベクトルになる。

 

.. math::  身長ベクトル = \begin{bmatrix} 170 \\\ 172 \\\ 175 \end{bmatrix} 

 

◼︎スカラー(Scalar)・・・ひとつひとつの要素（数字自体）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

身長ベクトルの中の一郎くんの身長スカラーは170cmとなる。

.. code:: ipython3

    heightVerctor = np.array([170, 172, 175])
    
    print("身長ベクトル: ", heightVerctor)
    print("一郎くんの身長スカラー: ", heightVerctor[0])


.. parsed-literal::

    身長ベクトル:  [170 172 175]
    一郎くんの身長スカラー:  170


◼︎行列 (Matrix) ・・・列または行がいくつか並べられたものを行列と言う。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

例えば一郎、二郎、三郎くんの身長ベクトルに体重を追加すると行列になる。

.. math::  三兄弟行列 = \begin{bmatrix} 170 & 60 \\\ 172 & 70 \\\ 175 & 90 \end{bmatrix} 

.. code:: ipython3

    brotherMatrix = np.array([[170, 172, 175], [60, 70, 90]])
    
    print("三兄弟行列:\n", brotherMatrix)


.. parsed-literal::

    三兄弟行列:
     [[170 172 175]
     [ 60  70  90]]


◼︎行列とベクトルの積・・・元のスカラー全ての影響をうけるように計算する。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a12であればaの12(いちにぃ)成分と読む。元の1から新たな2の成分となることを意味する。

.. math::  \begin{bmatrix} a_{11} & a_{12} & a_{13} \\\ a_{21} & a_{22} & a_{32}\\\ a_{13} & a_{23} & a_{33} \end{bmatrix} \begin{bmatrix} b_1 \\\ b_2 \\\ b_3 \end{bmatrix} =\begin{bmatrix} a_{11}b_1 + a_{21}b_2 + a_{31}b_3 \\\ a_{12}b_1 + a_{22}b_2 + a_{32}b_3 \\\ a_{13}b_1 + a_{23}b_2 + a_{33}b_3 \end{bmatrix} 

三兄弟行列にさらに年齢を追加して計算してみる。 

.. math::  \begin{bmatrix} 170 & 60 & 22 \\\ 172 & 70 & 21 \\\ 175 & 90 & 20 \end{bmatrix} \begin{bmatrix} 1 \\\ 2 \\\ 3 \end{bmatrix} =\begin{bmatrix} 170*1 + 172 * 2 + 175 * 3 \\\ 60*1 + 70*2 + 90*3 \\\ 22*1 + 21*2 + 20*3 \end{bmatrix} 

要素同士の積を足し合わせることで線形（直線っぽい）な結果を求めることができる。線形を求める代数（関数）なので、これらを線形代数という。

.. code:: ipython3

    brotherMatrixWithAge = np.array([[170, 172, 175], [60, 70, 90], [22, 21, 20]])
    multi = np.array([1, 2, 3])
    multiBrother = np.dot(brotherMatrixWithAge, multi)
    
    print("行列とベクトルの積 結果:\n", multiBrother)


.. parsed-literal::

    行列とベクトルの積 結果:
     [1039  470  124]


◼︎行列がかけられたベクトルはどんな変化をする？？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

あるベクトルにある行列をかけると（行列とベクトルの積）、ベクトルは回転して引き延ばされる！
試しに以下のようにx = [1,
1]に適当な行列をかけてみる。元々のベクトルは黒色、積の行列は青色で図示されている。黒いベクトルが青いベクトルに回転して引き延ばされていることがわかる。

.. code:: ipython3

    A = [[ 2, 1],
            [-1,-2]]
    x = [1, 1]
    
    a = np.dot(A, x)
    
    # 始点
    X_s = 0
    Y_s = 0
    # 終点
    X_e = x[0]
    Y_e = x[1]
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1)
    
    X_s = 0
    Y_s = 0
    X_e = a[0]
    Y_e = a[1]
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1, color='blue')
    
    # グラフ表示
    plt.xlim([-1,4])
    plt.ylim([-4,2])
    plt.grid()
    plt.draw()
    plt.show()



.. image:: rabbit_LinearAlgebra_files/rabbit_LinearAlgebra_9_0.png


◼︎行列同士の掛け算 （行列積）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::  \begin{bmatrix} a & b & c \\\ d & e & f \end{bmatrix} \begin{bmatrix} g & h \\\ i & j \\\ k & l \end{bmatrix} = \begin{bmatrix} ag + bi + ck & ah + bj + cl \\\ dg + ei + fk & dh + ej + fl \end{bmatrix}  

※Aの列とBの行が異なる場合は行列積ABは定義されない。

※一般的に AB != BA

.. code:: ipython3

    brotherMatrix = np.array([[170, 172, 175], [60, 70, 90]])
    multiOne = np.array([[1, 1], [1, 1],  [1, 1]])
    multiBrotherMatrix = np.dot(brotherMatrix, multiOne)
    
    print("行列積 結果:\n", multiBrotherMatrix)


.. parsed-literal::

    行列積 結果:
     [[517 517]
     [220 220]]


◼︎行列式・・・行列の特徴を表す指標（大きさみたいなもの）で、数（スカラー）で表す。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

例えば以下のような行列があるとする。 

.. math::  A = \begin{bmatrix} a & b \\\ c & d \end{bmatrix} 

これを行列式で表すと以下のようになる。

.. math::  \left| A \right| =\begin{vmatrix} a & b \\\ c & d \end{vmatrix} = det \begin{bmatrix} a & b \\\ c & d \end{bmatrix} = ad - bc 

3次元以上の行列式は以下のよう計算する。たすき掛けのように右向きのたすきは和、左向きのたすきは差として計算する。

.. math::  \left| A \right| =\begin{vmatrix} a_{11} & a_{12} & a_{13} \\\ a_{21} & a_{22} & a_{23}\\\ a_{31} & a_{32} & a_{33} \end{vmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32} 

.. code:: ipython3

    sample_matrix_data = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    
    print("行列式")
    print(linalg.det(sample_matrix_data))


.. parsed-literal::

    行列式
    -4.0


◼︎行列式のイメージ
~~~~~~~~~~~~~~~~~~

行列式は面積のイメージ。例えば上記の2x2の正方行列の場合はベクトルで囲まれる平行四辺形の面積と一致する。
以下の例は、ab = [[1, 2], [1, 1]] という行列の行列式は　a = [1, 2], b =
[1, 1]
の二つのベクトルで囲まれる平行四辺形の面積に一致する。（下図の黒いベクトルと青いベクトルで囲まれた面積。）

.. code:: ipython3

    plt.figure()
    
    # 始点
    X_s = 0, 0
    Y_s = 0, 0
    # 終点
    X_e = 1, 1
    Y_e = 2, 1
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1)
    
    X_s = 1, 1
    Y_s = 2, 1
    X_e = 1, 1
    Y_e = 1, 2
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1, color='blue')
    
    # グラフ表示
    plt.xlim([0,2])
    plt.ylim([0,3])
    plt.grid()
    plt.draw()
    plt.show()



.. image:: rabbit_LinearAlgebra_files/rabbit_LinearAlgebra_15_0.png


◼︎単位行列 I ・・・積が同じ行列になる行列
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::  I = \begin{bmatrix} 1 & 0 & 0 \\\ 0 & 1 & 0 \\\ 0 & 0 & ... \end{bmatrix} 

例は以下のような感じ。

.. math:: \begin{bmatrix} 1 & 0 \\\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 3 \\\ 2 & 4 \end{bmatrix} =\begin{bmatrix} 1 & 3 \\\ 2 & 4 \end{bmatrix} 

 

.. math:: \begin{bmatrix} 1 & 3 \\\ 2 & 4 \end{bmatrix} \begin{bmatrix} 1 & 0 \\\ 0 & 1 \end{bmatrix}  =\begin{bmatrix} 1 & 3 \\\ 2 & 4 \end{bmatrix} 

AB = I となる行列BをAの\ **逆行列**\ という（この時BA = Iも成り立つ）

逆行列をもつ行列Aを\ **正則行列**\ という

.. code:: ipython3

    sample_matrix_data = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    
    print("逆行列")
    print(linalg.inv(sample_matrix_data))
    
    print("元データと逆行列の積をとり、単位行列になっているか確認")
    print(sample_matrix_data.dot(linalg.inv(sample_matrix_data)))


.. parsed-literal::

    逆行列
    [[ 0.  -0.5 -0.5]
     [-0.5 -0.  -0.5]
     [-0.5 -0.5  0. ]]
    元データと逆行列の積をとり、単位行列になっているか確認
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]


◼︎固有ベクトル、固有値
~~~~~~~~~~~~~~~~~~~~~~

行列Aとベクトルxの積はある値λ(スカラー)とベクトルxの積と一致することがある。このベクトルxを\ **固有ベクトル**\ 、スカラーλを\ **固有値**\ を言う。行列Aの特徴や関係性を調べる際に用いられる。また、λxと表記する方が行列の積より簡単である。

.. math::  Ax = λx 

.. math::  (A -λI)x = 0  (ただし x != 0) 

.. math::  \left| A - λI \right| = 0 

例えば行列Aが以下のようだと、（Iは単位行列）

 

.. math::  A = \begin{bmatrix} 1 & 4 \\\ 2 & 3 \end{bmatrix} 

 

.. math::  \left| A - λI \right| = 0 

.. math::  \begin{vmatrix} 1 - λ & 4 \\\ 2 & 3 - λ \end{vmatrix} = 0 

.. math::  (1 - λ)(3 - λ) - 4 × 2 = 0 

.. math::  λ= 5 または -1 

これはつまり、

 

.. math::  \begin{bmatrix} 1 & 4 \\\ 2 & 3 \end{bmatrix} \begin{bmatrix} x_1 \\\ x_2 \end{bmatrix} = 5 \begin{bmatrix} x_1 \\\ x_2 \end{bmatrix} ・・・①

 

.. math::  \begin{bmatrix} 1 & 4 \\\ 2 & 3 \end{bmatrix} \begin{bmatrix} x_1 \\\ x_2 \end{bmatrix} = -1 \begin{bmatrix} x_1 \\\ x_2 \end{bmatrix} ・・・②

 

まず①の式(λ=5)について解くと、 

.. math::  \begin{bmatrix} x_1 + 4x_2 \\\ 2x_1 + 3x_2 \end{bmatrix} = \begin{bmatrix} 5x_1 \\\ 5x_2 \end{bmatrix} 

となり、以下の条件の時に\ **固有ベクトル**\ となる

 

.. math::  x_1 = x_2 

 

 

次に②の式(λ=-1)について解くと、  

.. math::  \begin{bmatrix} x_1 + 4x_2 \\\ 2x_1 + 3x_2 \end{bmatrix} = \begin{bmatrix} -x_1 \\\ -x_2 \end{bmatrix} 

となり、以下の条件の時に\ **固有ベクトル**\ となる

 

.. math::  x_1 = -2x_2 

 

 

λ=5の時、(1, 1)の定数倍。とかけるし

.. math::  \begin{bmatrix} 1 & 4 \\\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\\ 1 \end{bmatrix} =  5\begin{bmatrix} 1 \\\ 1 \end{bmatrix} 

 

λ=-1の時、(1, -1/2)の定数倍。ともかける。

 

.. math::  \begin{bmatrix} 1 & 4 \\\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\\ -1/2 \end{bmatrix} = -1 \begin{bmatrix} 1 \\\ -1/2 \end{bmatrix} 

関係性がわかりやすく、複雑な情報を単純に表せるようになった。

※ \|λI - A\| は λ
についてのn次多項式になり，これを\ **固有多項式**\ と言う

.. code:: ipython3

    sampleMatrix = np.array([[1, 2], [4, 3]])
    eig_value, eig_vector = linalg.eig(sampleMatrix)
    
    print("固有値")
    print(eig_value)
    print("固有ベクトル")
    print(eig_vector)


.. parsed-literal::

    固有値
    [-1.+0.j  5.+0.j]
    固有ベクトル
    [[-0.70710678 -0.4472136 ]
     [ 0.70710678 -0.89442719]]


◼︎固有ベクトルってどんなベクトル？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

固有ベクトルと、その行列をかけたベクトルは、必ず一直線になる。任意の行列とベクトルをかけると、ベクトルが回転して引き延ばされることは、これまでに試した。固有ベクトルは回転されず、直線に引き伸ばしたり逆方向のベクトルになり、一直線になることが大きな特徴である。

.. code:: ipython3

    # 始点
    X_s = 0
    Y_s = 0
    # 終点
    X_e = eig_vector[0,0]
    Y_e = eig_vector[1,0]
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1)
    
    MultiEigMatrix = np.dot(sampleMatrix, eig_vector)
    
    X_s = 0
    Y_s = 0
    X_e = MultiEigMatrix[0,0]
    Y_e = MultiEigMatrix[1,0]
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1, color='blue')
    
    # グラフ表示
    plt.xlim([-1,1])
    plt.ylim([-2,2])
    plt.grid()
    plt.draw()
    plt.show()



.. image:: rabbit_LinearAlgebra_files/rabbit_LinearAlgebra_21_0.png


.. code:: ipython3

    # 始点
    X_s = 0
    Y_s = 0
    # 終点
    X_e = eig_vector[0,1]
    Y_e = eig_vector[1,1]
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1)
    
    MultiEigMatrix = np.dot(sampleMatrix, eig_vector)
    
    X_s = 0
    Y_s = 0
    X_e = MultiEigMatrix[0,1]
    Y_e = MultiEigMatrix[1,1]
    
    plt.quiver(X_s,Y_s,X_e,Y_e,angles='xy',scale_units='xy',scale=1, color='blue')
    
    # グラフ表示
    plt.xlim([-4,1])
    plt.ylim([-8,2])
    plt.grid()
    plt.draw()
    plt.show()



.. image:: rabbit_LinearAlgebra_files/rabbit_LinearAlgebra_22_0.png


◼︎固有値分解
~~~~~~~~~~~~

正方形の行列を、"３つの行列の積"
に変換することを\ **固有値分解**\ と言う。

固有値を対角線上に並べた行列Λ（ラムダ）

.. math::  Λ = \begin{bmatrix} λ_1 & 0 & 0 \\\ 0 & λ_2 & 0 \\\ 0 & 0 & ... \end{bmatrix} 

それに対応する固有ベクトルを並べた行列V (λ1に対応するのはv1)

.. math::  V = \begin{bmatrix} v_1 & v_2 & ... \end{bmatrix} 

※λの順序は大きい順や小さい順に並べることが多いが順番で結果は変わらない。

に対して、正方行列Aは以下のように表される。

.. math::  AV = VΛ 

 

従って

.. math::  A = VΛV^{-1} 

 

[これまでに使った固有値と固有ベクトルを使った具体例]

.. math:: \begin{bmatrix} 1 & 4 \\\ 2 & 3 \end{bmatrix}  = \begin{bmatrix} 1 & 1 \\\ 1 & -1/2 \end{bmatrix} \begin{bmatrix} 5 & 0 \\\ 0 & -1 \end{bmatrix} \begin{bmatrix} 1/3 & 2/3 \\\ 2/3 & -2/3 \end{bmatrix} 

※
Vは定数倍でも良いので別の書き方もできるが、その時は最後の逆行列も変更が必要。

◼︎実際に固有値分解をやってみる
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

まず以下の固有値と固有ベクトルを求める。

.. math::  A = \begin{bmatrix} -2 & 1 & -1 \\\ 1 & -2 & 1 \\\ -1 & 1 & -2 \end{bmatrix} 

余因子分解で固定値と固定ベクトルを求め、掃き出し法で逆行列を求めたら固有値分解の完了。(書くと大変なので省略、また今後メンテする)

結果、固有値 -4 (重複度1), 固有値 -1 (重複度2)
になり、固有ベクトルwは以下のようになる。

.. math::  w_{-4} = α \begin{bmatrix} 1 \\\ -1 \\\ 1 \end{bmatrix} , α != 0

.. math::  w_{-1} = β \begin{bmatrix} 1 \\\ 1 \\\ 0 \end{bmatrix} + γ \begin{bmatrix} -1 \\\ 0 \\\ 1 \end{bmatrix} , (β, γ) != 0

.. math::  この時、　(w_{-4}, w_{-1}) = 0 を満たす。

.. math::  \begin{bmatrix} -2 & 1 & -1 \\\ 1 & -2 & 1 \\\ -1 & 1 & -2 \end{bmatrix} = \begin{bmatrix} 1 & 1 & -1 \\\ -1 & 1 & 0 \\\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} -4 & 0 & 0 \\\ 0 & -1 & 0 \\\ 0 & 0 & -1 \end{bmatrix} \begin{bmatrix} -3/4 & -1/4 & 1/4 \\\ -1/4 & -3/4 & -1/4 \\\ 1/4 & -1/4 & -3/4 \end{bmatrix} 

.. code:: ipython3

    # 答え合わせ
    V = np.array([[1, -1, 1], [1, 1, 0], [-1, 0, 1]]).T
    E = np.diagflat(np.array([-4, -1, -1]))
    V1 = np.linalg.inv(V)
    np.dot((np.dot(V, E)), V1)




.. parsed-literal::

    array([[-2.,  1., -1.],
           [ 1., -2.,  1.],
           [-1.,  1., -2.]])



.. code:: ipython3

    # pythonのライブラリ使うと固有値分解って簡単に求まるねぇ
    A = np.array([[-2, 1, -1], [1, -2, 1], [-1, 1, -2]])
    
    eig_value, eig_vector = linalg.eig(A)
    
    print("元のベクトル")
    print(A)
    print("固有ベクトル")
    print(eig_vector)
    print("固有値")
    print(np.diagflat(eig_value))
    print("固有ベクトルの逆行列")
    print(np.linalg.inv(eig_vector))
    
    np.dot((np.dot(eig_vector, np.diagflat(eig_value))), np.linalg.inv(eig_vector))


.. parsed-literal::

    元のベクトル
    [[-2  1 -1]
     [ 1 -2  1]
     [-1  1 -2]]
    固有ベクトル
    [[ 0.81649658  0.57735027  0.381008  ]
     [ 0.40824829 -0.57735027  0.81590361]
     [-0.40824829  0.57735027  0.43489561]]
    固有値
    [[-1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j -4.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j -1.+0.j]]
    固有ベクトルの逆行列
    [[ 8.16496581e-01  3.51767490e-02 -7.81319832e-01]
     [ 5.77350269e-01 -5.77350269e-01  5.77350269e-01]
     [ 4.43805449e-17  7.99488821e-01  7.99488821e-01]]




.. parsed-literal::

    array([[-2.+0.j,  1.+0.j, -1.+0.j],
           [ 1.+0.j, -2.+0.j,  1.+0.j],
           [-1.+0.j,  1.+0.j, -2.+0.j]])



◼︎特異値分解
~~~~~~~~~~~~

・例えば１０００個の要素を扱う時に特異値分解すると、とても小さい数を扱えばよくなる。

・正方行列以外に対して行う固有値分解のことを特異値分解と言う。

以下、Mという正方行列でない行列がある時、右特異ベクトルvとの積によってある特異値σをつかってσuという値が求まるとする。この左特異ベクトルuの成分の数はvとは異なる。

.. math::  Mv = σu 

これはMと転置すると、Mとの掛け算は、σvと求めることができる。

.. math::  M^Tu = σv 

このような特殊な単位ベクトルがあるならば特異値分解できる。

.. math::  M = USV^{-1} 

.. math::  V ・・・ v_1 .... v_n 

.. math::  U ・・・ u_1 .... u_n 

.. math::  S ・・・ σ_1 .... σ_n 

◼︎特異値の求め方
~~~~~~~~~~~~~~~~

下記の式が成り立つことは、上記のMv = σuが成り立つことからわかる。

.. math::  MV = US 

.. math::  M = USV^{-1} 

また、下記の式が成り立つことも転置の式からわかる。

.. math::  M^TU = VS^T 

.. math::  M^T = VSU^{-1} 

これらの積（長方形行列×その転置＝正方行列）は固有値分解可能である。

.. math::  MM^T = USV^{-1}VSU^{-1} = USS^TU^{-1} 

特異値分解は、無理やり正方行列を作り、それを固有値分解したことからはじまっている。

.. math::  SS^T 

というのは、特異値の二乗がずらーっと並んだ行列である。つまり、MとMの転置の積で出てくる正方行列を固有値分解して出てくる固有値Λ行列は、特異値の二乗になっている。従って、求まった固有値Λ行列の成分の√をとると、特異値S行列が求まることがわかる。

.. code:: ipython3

    # 特異値ベクトルを解いてみよう！！
    # linalg.eig で求められる固有ベクトルは単位行列になるよう計算されて出力される
    
    M = np.array([[1, 2, 3], [3, 2, 1]])
    
    M_eig = np.dot(M, M.T)
    print ("MM^T固有分解前\n", M_eig)
    S, U = linalg.eig(M_eig)
    print ("固有ベクトルV\n", U)
    print ("固有値Λ\n", np.diagflat(S))
    print ("固有ベクトルの逆行列\n", np.linalg.inv(U))
    M_eig = np.dot(np.dot(U, np.diagflat(S)), np.linalg.inv(U))
    print ("固有分解の確認\n", M_eig)
    
    M_eig = np.dot(M.T, M)
    print ("M^T M固有分解前\n", M_eig)
    S, V = linalg.eig(M_eig)
    print ("固有ベクトルV\n", V)
    print ("固有値Λ\n", np.diagflat(S))
    print ("固有ベクトルの逆行列\n", np.linalg.inv(V))
    M_eig = np.dot(np.dot(V, np.diagflat(S)), np.linalg.inv(V))
    print ("固有分解の確認\n", M_eig)
    
    # 答え合わせ
    # √にして固有値から特異値を求める。
    S = np.array([[np.sqrt(S[0]), 0, 0], [0, np.sqrt(S[1]), 0]])
    print ("左特異ベクトルU\n", U)
    print ("特異値S\n", S)
    print ("右特異ベクトルV^{-1}\n", np.linalg.inv(V))
    
    # ライブラリも使って答え合わせ
    svd_U, svd_S, svd_V = linalg.svd(M)
    #　linalg.svd出力の特異値はベクトルのため、専用の linalg.diagsvd で行列に変換
    svd_S = linalg.diagsvd(svd_S, 2, 3)
    print ("ライブラリで求めた左特異ベクトルU\n", svd_U)
    print ("ライブラリで求めた特異値S\n", svd_S)
    print ("ライブラリで求めた右特異ベクトルV^{-1}\n", svd_V)
    print ("ライブラリの確認\n", np.dot(np.dot(svd_U, svd_S), svd_V))
    
    M = np.dot(np.dot(U, S), np.linalg.inv(V))
    M


.. parsed-literal::

    MM^T固有分解前
     [[14 10]
     [10 14]]
    固有ベクトルV
     [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]
    固有値Λ
     [[24.+0.j  0.+0.j]
     [ 0.+0.j  4.+0.j]]
    固有ベクトルの逆行列
     [[ 0.70710678  0.70710678]
     [-0.70710678  0.70710678]]
    固有分解の確認
     [[14.+0.j 10.+0.j]
     [10.+0.j 14.+0.j]]
    M^T M固有分解前
     [[10  8  6]
     [ 8  8  8]
     [ 6  8 10]]
    固有ベクトルV
     [[-5.77350269e-01 -7.07106781e-01  4.08248290e-01]
     [-5.77350269e-01 -5.84963758e-16 -8.16496581e-01]
     [-5.77350269e-01  7.07106781e-01  4.08248290e-01]]
    固有値Λ
     [[ 2.40000000e+01+0.j  0.00000000e+00+0.j  0.00000000e+00+0.j]
     [ 0.00000000e+00+0.j  4.00000000e+00+0.j  0.00000000e+00+0.j]
     [ 0.00000000e+00+0.j  0.00000000e+00+0.j -1.92395977e-15+0.j]]
    固有ベクトルの逆行列
     [[-5.77350269e-01 -5.77350269e-01 -5.77350269e-01]
     [-7.07106781e-01  9.61481343e-17  7.07106781e-01]
     [ 4.08248290e-01 -8.16496581e-01  4.08248290e-01]]
    固有分解の確認
     [[10.+0.j  8.+0.j  6.+0.j]
     [ 8.+0.j  8.+0.j  8.+0.j]
     [ 6.+0.j  8.+0.j 10.+0.j]]
    左特異ベクトルU
     [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]
    特異値S
     [[4.89897949+0.j 0.        +0.j 0.        +0.j]
     [0.        +0.j 2.        +0.j 0.        +0.j]]
    右特異ベクトルV^{-1}
     [[-5.77350269e-01 -5.77350269e-01 -5.77350269e-01]
     [-7.07106781e-01  9.61481343e-17  7.07106781e-01]
     [ 4.08248290e-01 -8.16496581e-01  4.08248290e-01]]
    ライブラリで求めた左特異ベクトルU
     [[-0.70710678 -0.70710678]
     [-0.70710678  0.70710678]]
    ライブラリで求めた特異値S
     [[4.89897949 0.         0.        ]
     [0.         2.         0.        ]]
    ライブラリで求めた右特異ベクトルV^{-1}
     [[-5.77350269e-01 -5.77350269e-01 -5.77350269e-01]
     [ 7.07106781e-01  3.05311332e-16 -7.07106781e-01]
     [ 4.08248290e-01 -8.16496581e-01  4.08248290e-01]]
    ライブラリの確認
     [[1. 2. 3.]
     [3. 2. 1.]]




.. parsed-literal::

    array([[-1.+0.j, -2.+0.j, -3.+0.j],
           [-3.+0.j, -2.+0.j, -1.+0.j]])



◼︎特異値分解を使って画像を圧縮してみよう
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

特異値分解を使って画像データの行列から成分の小さい部分を取り除く。これを\ **低ランク近似**\ と言う。原理としては、左特異ベクトルu,
特異値σ, 右特異ベクトルv を使った以下の式を用いる。

.. math::  A_k = \sum _{i=1} ^{k} {u_i σ_i v_i}, k = 1, 2, ... max(m, n) 

.. code:: ipython3

    # まず、元画像をグレースケール
    image = Image.open('dog.jpg')
    gray_image = image.convert('L')
    gray_image.save('mono_dog.jpg')
    print("元画像")
    gray_image


.. parsed-literal::

    元画像




.. image:: rabbit_LinearAlgebra_files/rabbit_LinearAlgebra_29_1.png



.. code:: ipython3

    gray_image = Image.open('mono_dog.jpg')
    gray_image_array = np.array(gray_image)
    
    #  特異値分解
    svd_U, svd_S, svd_V = linalg.svd(gray_image_array)
    
    # 低ランク近似値
    k = 32
    
    # 左特異行列Uからk列目までの左特異ベクトルを抜き出す/ shape [853, 32]
    U = svd_U[:, :k]
    # 特異値ベクトルを特異値行列へ変換し、近似値分抜き出す/ shape [32, 32]
    S = np.matrix(linalg.diagsvd(svd_S[:k], k, k))
    # 右特異ベクトルVからk行目までの右特異ベクトルを抜き出す / shape [32, 1280]
    V = svd_V[:k, :]
    
    # 注意：行列積ではなく、要素同士の積を計算する
    new_image = U*S*V
    new_image = Image.fromarray(np.uint8(new_image))
    #new_image.save('k32_mono_dog.jpg')
    print("特異値分解による画像圧縮後")
    new_image


.. parsed-literal::

    特異値分解による画像圧縮後




.. image:: rabbit_LinearAlgebra_files/rabbit_LinearAlgebra_30_1.png



.. code:: ipython3

    # Python スライスの使い方
    sample = np.array([[11,12,13], [21,22,23]])
    print("スライスで1列目までを表示\n", sample[:, :1])
    print("スライスで2列目までを表示\n", sample[:, :2])


.. parsed-literal::

    スライスで1列目までを表示
     [[11]
     [21]]
    スライスで2列目までを表示
     [[11 12]
     [21 22]]


◼︎補足
~~~~~~

ベクトルの内積
^^^^^^^^^^^^^^

.. math::  x = \begin{bmatrix} x_1 & x_2 & ... &  x_n \end{bmatrix} 

.. math::  y = \begin{bmatrix} y_1 & y_2 & ... &  y_n \end{bmatrix} 

これらxベクトルとyベクトルの内積は、

.. math::  x_1y_1 + x_2y_2 + ... + x_ny_n 

※ 次元が違うベクトル同士の内積は定義されない

零行列
^^^^^^

全ての要素が０の行列を０行列と言う

逆行列
^^^^^^

AB = I となる行列BをAの逆行列と言い、

.. math::  B = A^{-1} 

※逆行列は存在しないこともある。逆行列は余因子を使って表せる行列を行列式で割ったものなので、行列式が0である場合は、その行列は逆行列を持たないことになる。

.. math::  ｜A｜＝ad-bc=0 

となる行列は逆行列を持たない。

転置
^^^^

.. math::  （A^T)_{ij} = A_ji 

.. math::  \begin{bmatrix} a & b & c \\\ d & e & f \end{bmatrix}^T = \begin{bmatrix} a & d \\\ b & e \\\ c & f \end{bmatrix} 

となる

.. math::  A^T = A 

となる正方行列Aを対称行列という

余因子展開
^^^^^^^^^^

.. math::  \begin{bmatrix} a & b & c \\\ d & e & f \\\ g & h & i \end{bmatrix} = a \begin{vmatrix} e & f \\\ h & i \end{vmatrix} - b \begin{vmatrix} d & f \\\ g & i \end{vmatrix} + c \begin{vmatrix} d & e \\\ g & h \end{vmatrix} 

重複度
^^^^^^

代数学の基本定理
n次方程式は複素数の範囲に(重複度を含めて)必ずn個の解を持つ。というのが「代数学の基本定理」であった。固有値と固有ベクトルの式、det(A-λI)=0　はλについてのn次多項式となる。

.. math::  σ_A(t) = (t-λ_1)^{n_1} .... (t-λ_m)^{n_m} 

自然数 n(i) のことをλ(i) の\ **重複度**\ という

実対称行列
^^^^^^^^^^

.. math::  A = A^T 

特徴：　固有値は全て実数。対角化可能。異なる固有値に対する固有ベクトルは直行する（実対象行列は直行行列を用いて対角化可能）

単位ベクトル
^^^^^^^^^^^^

長さが1のベクトルを単位ベクトルと言う。ベクトルaと同じ向きの単位ベクトルは
a/\|a\| で求まる。

a = (3, 4)と同じ向きの単位ベクトルは、

.. math::  |a| = \sqrt{3^2+3^2} = 5 

aの単位ベクトル = (3/5, 4/5)

◼︎引用, 参考資料
~~~~~~~~~~~~~~~~

・ラビットチャレンジ - 応用数学講座

http://ai999.careers/rabbit/

・東京大学グローバル消費インテリジェンス寄付講座 - Data Science Online
Course

https://gci.t.u-tokyo.ac.jp/

・京都大学講義資料 - ビッグデータの計算科学

http://www.iedu.i.kyoto-u.ac.jp/uploads/20141022.pdf

・asta muse - 特異値分解と行列の低ランク近似

http://lab.astamuse.co.jp/entry/2017/06/14/114500

・Qiita - 特異値分解による画像の低ランク近似

https://qiita.com/kaityo256/items/48de63526b469235d16a

・Qiita - 【数学】固有値・固有ベクトルとは何かを可視化してみる

https://qiita.com/kenmatsu4/items/2a8573e3c878fc2da306

・基礎数学ワークブック - 平面ベクトルと行列式

http://www.core.kochi-tech.ac.jp/m\_inoue/work/pdf/2006/syokyu07/18.pdf

・高校数学の基本問題 - 固有値, 固有ベクトルの求め方

http://www.geisya.or.jp/~mwm48961/linear\_algebra/eigenvalue2.htm
