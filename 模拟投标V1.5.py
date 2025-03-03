import streamlit as st
import pandas as pd
import numpy as np
from itertools import product

# ----------------------------
# 评分规则定义
# ----------------------------
#评分规则1
def rule_avg_based(prices, my_price):
    """
    基于平均报价计算得分，报价越接近均价，得分越高。

    参数:
    prices (list): 所有投标人的报价列表
    my_price (float): 当前投标人的报价

    返回:
    float: 当前投标人的得分
    """
    avg = sum(prices) / len(prices)
    deviation = abs(my_price - avg) / avg
    return max(0, 100 - deviation * 1000)  # 示例公式，可根据实际规则调整

#评分规则2
def rule_lowest_based(prices, my_price):
    """
    以最低报价为基准计算得分，报价最低者得100分，其他报价按与最低报价的差值计分。

    参数:
    prices (list): 所有投标人的报价列表
    my_price (float): 当前投标人的报价

    返回:
    float: 当前投标人的得分
    """
    min_price = min(prices)
    return 100 if my_price == min_price else max(0, 100 - (my_price - min_price) / min_price * 100)

#评分规则3
def new_rule(prices, my_price, K):
    """
    采用区间中间值法计算价格得分。

    参数:
    prices (list): 所有投标人的报价列表
    my_price (float): 当前投标人的报价
    K (float): 区间系数，范围为 0.9 - 0.99

    返回:
    float: 当前投标人的价格得分，保留两位小数
    """
    E = 1
    # 计算评标基准价 D
    if len(prices) <= 5:
        # 有效投标人数量 ≤ 5，D 为有效投标人评标价格的算术平均值
        D = sum(prices) / len(prices)
    else:
        # 有效投标人数量 > 5，先剔除 1 个最高和 1 个最低评标价格
        sorted_prices = sorted(prices)
        temp_prices = sorted_prices[1:-1]
        # 计算剩余价格的算术平均值
        avg_temp = sum(temp_prices) / len(temp_prices)
        final_prices = []
        # 剔除与平均值偏差 20% 以上的价格
        for price in temp_prices:
            deviation = abs(price - avg_temp) / avg_temp
            if deviation <= 0.2:
                final_prices.append(price)
        # 若所有剩余价格偏差均在 20% 以上，则不做剔除
        if not final_prices:
            final_prices = temp_prices
        # 最终 D 为剩余有效投标人评标价格的算术平均值
        D = sum(final_prices) / len(final_prices)

    # 根据评标价格与评标基准价的关系计算得分
    if my_price == D * K:
        F = 100
    elif my_price > D * K:
        F = 100 - abs(my_price - D * K) / (D * K) * 100 * E
    else:
        F = 100 - abs(my_price - D * K) / (D * K) * 100 * (E / 2)
    # 得分最低为 0
    F = max(0, F)
    return round(F, 2)

#评分规则4
def rule_insert_method(prices, my_price):
    """
    采用插入法计算得分。
    参数:
    prices (list): 所有投标人的报价列表
    my_price (float): 当前投标人的报价

    返回:
    float: 当前投标人的得分，保留两位小数
    """
    if len(prices) < 5:
        # 投标人数小于5个，评标价为所有投标报价算术平均值
        P = sum(prices) / len(prices)
    else:
        # 投标人数大于等于5个，去掉1个最高和1个最低投标报价后求平均值
        sorted_prices = sorted(prices)
        temp_prices = sorted_prices[1:-1]
        P = sum(temp_prices) / len(temp_prices)

    # 计算偏差率
    deviation_rate = round(100 * (my_price - P) / P, 2)

    # 按插入法计算得分
    if my_price > P:
        score = max(0, 50 - abs(deviation_rate))
    elif my_price == P:
        score = 50
    else:
        score = max(0, 50 - abs(deviation_rate) * 0.5)

    return round(score, 2)

def rule_improved_insert_method(prices, my_price, F, a, b):
    """
    改进版插入法评分。

    参数:
    prices (list): 所有投标人的报价列表
    my_price (float): 当前投标人的报价
    F (float): 投标报价部分满分
    a (float): 投标报价每高于评标基准价P一个百分点的扣分数
    b (float): 投标报价每低于评标基准价P一个百分点的扣分数

    返回:
    float: 当前投标人的得分，保留两位小数
    """
    M = len(prices)
    if M <= 5:
        n = 0
    elif 5 < M <= 10:
        n = 1
    elif 10 < M <= 20:
        n = 2
    else:
        n = 3
    sorted_prices = sorted(prices)
    remaining_prices = sorted_prices[n:M - n]
    P = round(sum(remaining_prices) / len(remaining_prices), 2)

    if my_price <= P:
        Sx = F - (P - my_price) * b / P
    else:
        Sx = F - (my_price - P) * a / P
    Sx = max(0, Sx)
    return round(Sx, 2)

def create_result_df(unit_types, all_discounts, all_prices, scores, my_score):
    """
    创建包含投标结果的 DataFrame。

    参数:
    unit_types (list): 单位类型列表，如 ['我方', '队友', '对手']
    all_discounts (list): 所有投标人的折扣列表
    all_prices (list): 所有投标人的报价列表
    scores (list): 所有投标人的得分列表
    my_score (float): 我方的得分

    返回:
    pandas.DataFrame: 包含投标结果的 DataFrame
    """
    df = pd.DataFrame({
        '单位类型': unit_types,
        '投标折扣（%）': all_discounts,
        '报价': all_prices,
        '得分': scores
    })
    df['与我方得分分差'] = df['得分'].apply(lambda x: round(x - my_score, 2))
    df['排名'] = df['得分'].rank(ascending=False, method='min').astype(int)

    # 添加样式标记列（用于后续样式应用）
    df['_style'] = df['单位类型'].apply(lambda x:
                                        'my_row' if x == '我方' else
                                        'teammate_row' if x == '队友' else
                                        'opponent_row'
                                        )

    # 分别处理我方、队友和对手数据
    my_df = df[df['单位类型'] == '我方']
    teammate_df = df[df['单位类型'] == '队友']
    opponent_df = df[df['单位类型'] == '对手'].sort_values(by='排名')

    # 按顺序合并数据
    df = pd.concat([my_df, teammate_df, opponent_df]).reset_index(drop=True)

    # 添加序号列（从1开始）
    df.insert(0, '序号', df.index + 1)

    return df


# ----------------------------
# 投标模拟类
# ----------------------------
class BidSimulator:
    def __init__(self, rule_func, budget=100, price_weight=100):
        """
        初始化投标模拟类。

        参数:
        rule_func (function): 评分规则函数
        budget (float): 最高限价
        price_weight (float): 价格分比重，范围为 0 - 100
        """
        self.rule = rule_func
        self.budget = budget
        self.price_weight = price_weight / 100  # 将百分比转换为小数
        self.competitors = []
        self.teammates = []
        self.competitor_discounts = []
        self.teammate_discounts = []
        # 其他方法保持不变

    def simulate(self, my_price, my_discount, teammate_discounts=None, K=None, F=None, a=None, b=None):
        # 上面定义的 simulate 方法内容
        pass
    # 队友报价和折扣输入部分
    def generate_teammates(self, num=0):
        def discount_to_price(discount, budget):
            return discount / 100 * budget

        def price_to_discount(price, budget):
            return price / budget * 100

        self.teammates = []
        self.teammate_discounts = []
        for i in range(num):
            # 在小部件实例化之前初始化 st.session_state
            if f'teammate_discount_{i}' not in st.session_state:
                st.session_state[f'teammate_discount_{i}'] = 0.0
            if f'teammate_price_{i}' not in st.session_state:
                st.session_state[f'teammate_price_{i}'] = discount_to_price(0.0, self.budget)

            col1, col2 = st.columns(2)
            with col1:
                teammate_discount = st.number_input(
                    f"队友 {i + 1} 折扣（%）",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    key=f"teammate_discount_{i}",
                    on_change=lambda idx=i: st.session_state.update(
                        {f"teammate_price_{idx}": discount_to_price(st.session_state[f"teammate_discount_{idx}"],
                                                                    self.budget)})
                )
            with col2:
                teammate_price = st.number_input(
                    f"队友 {i + 1} 报价（万元）",
                    min_value=0.0,
                    max_value=self.budget,
                    step=0.01,
                    key=f"teammate_price_{i}",
                    on_change=lambda idx=i: st.session_state.update(
                        {f"teammate_discount_{idx}": price_to_discount(st.session_state[f"teammate_price_{idx}"],
                                                                       self.budget)})
                )
            self.teammate_discounts.append(st.session_state[f'teammate_discount_{i}'])
            self.teammates.append(st.session_state[f'teammate_price_{i}'])

    # 对手报价和折扣输入部分
    def generate_competitors(self, num=5):
        def discount_to_price(discount, budget):
            return discount / 100 * budget

        def price_to_discount(price, budget):
            return price / budget * 100

        self.competitors = []
        self.competitor_discounts = []
        for i in range(num):
            # 在小部件实例化之前初始化 st.session_state
            if f'competitor_discount_{i}' not in st.session_state:
                st.session_state[f'competitor_discount_{i}'] = 0.0
            if f'competitor_price_{i}' not in st.session_state:
                st.session_state[f'competitor_price_{i}'] = discount_to_price(0.0, self.budget)

            col1, col2 = st.columns(2)
            with col1:
                competitor_discount = st.number_input(
                    f"对手 {i + 1} 折扣（%）",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    key=f"competitor_discount_{i}",
                    on_change=lambda idx=i: st.session_state.update(
                        {f"competitor_price_{idx}": discount_to_price(st.session_state[f"competitor_discount_{idx}"],
                                                                      self.budget)})
                )
            with col2:
                competitor_price = st.number_input(
                    f"对手 {i + 1} 报价（万元）",
                    min_value=0.0,
                    max_value=self.budget,
                    step=0.01,
                    key=f"competitor_price_{i}",
                    on_change=lambda idx=i: st.session_state.update(
                        {f"competitor_discount_{idx}": price_to_discount(st.session_state[f"competitor_price_{idx}"],
                                                                         self.budget)})
                )
            self.competitor_discounts.append(st.session_state[f'competitor_discount_{i}'])
            self.competitors.append(st.session_state[f'competitor_price_{i}'])

    def simulate(self, my_price, my_discount, teammate_discounts=None, K=None, F=None, a=None, b=None):
        """
        运行投标模拟。

        参数:
        my_price (float): 我方的报价
        my_discount (float): 我方的折扣
        teammate_discounts (list): 队友的折扣列表
        K (float): 区间系数，仅在使用区间中间值法评分时需要
        F (float): 投标报价部分满分，仅在使用改进版插入法评分时需要
        a (float): 投标报价每高于评标基准价P一个百分点的扣分数，仅在使用改进版插入法评分时需要
        b (float): 投标报价每低于评标基准价P一个百分点的扣分数，仅在使用改进版插入法评分时需要

        返回:
        pandas.DataFrame: 包含模拟结果的 DataFrame
        """
        if teammate_discounts is None:
            teammate_discounts = self.teammate_discounts
        teammates = [discount / 100 * self.budget for discount in teammate_discounts]
        all_prices = [my_price] + teammates + list(self.competitors)
        all_discounts = [my_discount] + teammate_discounts + self.competitor_discounts
        unit_types = ['我方'] + ['队友'] * len(teammates) + ['对手'] * len(self.competitors)
        if self.rule == new_rule:
            scores = [self.rule(all_prices, p, K) for p in all_prices]
        elif self.rule == rule_improved_insert_method:
            scores = [self.rule(all_prices, p, F, a, b) for p in all_prices]
        else:
            scores = [self.rule(all_prices, p) for p in all_prices]

        # 将得分乘以价格分比重
        scores = [score * self.price_weight for score in scores]

        my_score = scores[0]
        df = create_result_df(unit_types, all_discounts, all_prices, scores, my_score)
        return df

    # 模式二：推荐队友折扣策略底层逻辑
    def recommend_teammate_discounts(self, my_price, my_discount, competitor_discounts, num_teammates, K=None, F=None,
                                     a=None, b=None):
        """
        推荐能使我方报价得分最高的队友折扣。

        参数:
        my_price (float): 我方的报价
        my_discount (float): 我方的折扣
        competitor_discounts (list): 对手的折扣列表
        num_teammates (int): 队友数量
        K (float): 区间系数，仅在使用区间中间值法评分时需要

        返回:
        list: 推荐的队友折扣列表
        """
        best_score = -np.inf
        best_discounts = []
        step = 1  # 折扣遍历步长
        discount_range = np.arange(0, 101, step)
        # 生成所有可能的队友折扣组合
        all_combinations = product(discount_range, repeat=num_teammates)
        for combination in all_combinations:
            teammate_discounts = list(combination)
            teammates = [discount / 100 * self.budget for discount in teammate_discounts]
            competitors = [discount / 100 * self.budget for discount in competitor_discounts]
            all_prices = [my_price] + teammates + competitors
            if self.rule == rule_improved_insert_method:
                scores = [self.rule(all_prices, p, F, a, b) for p in all_prices]
            elif self.rule == new_rule:
                scores = [self.rule(all_prices, p, K) for p in all_prices]
            else:
                scores = [self.rule(all_prices, p) for p in all_prices]
            scores = [score * self.price_weight for score in scores]
            my_score = scores[0]
            if my_score > best_score:
                best_score = my_score
                best_discounts = teammate_discounts
        return best_discounts



    # 模式三：推荐我方折扣策略底层逻辑
    def recommend_my_discount(self, teammate_discounts, competitor_discounts, K=None, F=None, a=None, b=None):
        """
        推荐使我方得分比任一对手高的最高折扣，若无，则推荐使我方得分最高的折扣

        参数:
        teammate_discounts (list): 队友折扣列表
        competitor_discounts (list): 对手折扣列表
        K (float): 区间系数

        返回:
        float: 推荐的我方折扣（%）
        """
        best_discount_over_opponents = 0.0
        best_score_over_opponents = -np.inf
        best_discount_global = 0.0
        best_score_global = -np.inf

        # 缩小折扣遍历步长，更精细地搜索
        step = 0.1
        for discount in np.arange(100.0, -step, -step):
            my_price = discount / 100 * self.budget
            teammates = [d / 100 * self.budget for d in teammate_discounts]
            competitors = [d / 100 * self.budget for d in competitor_discounts]
            all_prices = [my_price] + teammates + competitors

            try:
                if self.rule == rule_improved_insert_method:
                    scores = [self.rule(all_prices, p, F, a, b) for p in all_prices]
                elif self.rule == new_rule:
                    scores = [self.rule(all_prices, p, K) for p in all_prices]
                else:
                    scores = [self.rule(all_prices, p) for p in all_prices]

                scores = [s * self.price_weight for s in scores]
                my_score = scores[0]
                competitor_scores = scores[len(teammates) + 1:]

                # 检查是否全局最高得分
                if my_score > best_score_global:
                    best_score_global = my_score
                    best_discount_global = discount

                # 检查我方得分是否比任一对手高
                if all(my_score > score for score in competitor_scores):
                    if my_score > best_score_over_opponents:
                        best_score_over_opponents = my_score
                        best_discount_over_opponents = discount

            except Exception as e:
                print(f"计算得分时出现错误，折扣: {discount:.3f}%, 错误信息: {e}")

        # 如果找到了使我方得分高于所有对手的折扣，返回该折扣，否则返回全局最高得分对应的折扣
        if best_score_over_opponents != -np.inf:
            return round(best_discount_over_opponents, 2)
        else:
            return round(best_discount_global, 2)
    #模式四：推荐我方和队友策略底层逻辑
    def recommend_my_and_teammates_discounts(self, competitor_discounts, num_teammates, K=None, F=None, a=None, b=None):
        best_my_discount = 0.0
        best_teammate_discounts = []
        best_score = -np.inf
        step_my = 0.5  # 优化步长设置

        competitors = [d / 100 * self.budget for d in competitor_discounts]

        # 增加进度条显示
        progress_bar = st.progress(0)
        total_steps = int((100.0 - 0.0) / step_my)

        for idx, my_discount in enumerate(np.arange(100.0, -step_my, -step_my)):
            my_discount = round(my_discount, 1)
            my_price = my_discount / 100 * self.budget

            # 推荐当前我方折扣下的最佳队友折扣
            teammate_discounts = self.recommend_teammate_discounts(
                my_price, my_discount,
                competitor_discounts,
                num_teammates,
                K, F, a, b
            )

            # 计算得分
            teammates = [d / 100 * self.budget for d in teammate_discounts]
            all_prices = [my_price] + teammates + competitors

            try:  # 添加异常处理
                if self.rule == rule_improved_insert_method:
                    scores = [self.rule(all_prices, p, F, a, b) for p in all_prices]
                elif self.rule == new_rule:
                    scores = [self.rule(all_prices, p, K) for p in all_prices]
                else:
                    scores = [self.rule(all_prices, p) for p in all_prices]

                scores = [s * self.price_weight for s in scores]
                current_score = scores[0]

                # 更新最佳结果（得分更高或得分相同但折扣更高）
                if (current_score > best_score) or (current_score == best_score and my_discount > best_my_discount):
                    best_score = current_score
                    best_my_discount = my_discount
                    best_teammate_discounts = teammate_discounts

            except Exception as e:
                st.warning(f"折扣{my_discount}%计算异常: {str(e)}")

            # 更新进度条
            progress_bar.progress(min((idx + 1) / total_steps, 1.0))

        progress_bar.empty()
        return best_my_discount, best_teammate_discounts

# ----------------------------
# Streamlit界面
# ----------------------------
st.title("投标策略模拟分析系统")
st.markdown("版权归中国移动通信集团设计院所有")
# 侧边栏配置参数
with st.sidebar:
    st.header("模拟参数")
    budget = st.number_input("最高限价（万元）", value=1000.0)
    num_teammates = st.slider("队友数量", 0, 10, 1)
    num_competitors = st.slider("对手数量", 0, 10, 1)
    # 添加价格分比重输入框
    price_weight = st.number_input("价格分比重（%）", min_value=0.0, max_value=100.0, step=0.01, value=100.0)

    # 选择模式
    mode = st.selectbox("选择模式", ["模式一：手动填写各方报价", "模式二：自动推荐队友折扣", "模式三：自动推荐我方折扣", "模式四：推荐我方和队友折扣"])

    rule_dict = {
        "1: 平均报价评分": rule_avg_based,
        "2: 最低报价评分": rule_lowest_based,
        "3: 区间中间值法评分": new_rule,
        "4: 插入法计算得分": rule_insert_method,
        "5: 改进版插入法评分": rule_improved_insert_method
    }

    # 在 Streamlit 侧边栏的规则选择中更新选项
    rule_choice = st.selectbox("评分规则", list(rule_dict.keys()))

    # 在规则详情的 expander 中添加评分规则5的说明
    with st.expander("规则详情", expanded=True):
        st.markdown("### 规则说明")
        K = None
        F = None
        a = None
        b = None
        if rule_choice == "1: 平均报价评分":
            st.markdown("报价接近均价得分高。")
        elif rule_choice == "2: 最低报价评分":
            st.markdown("报价最低得100分，否则按差值计分。")
        elif rule_choice == "3: 区间中间值法评分":
            st.markdown("""
            价格采用中间值法中的区间中间值法，具体计分方法如下:
            1. 评标价格等于评标基准价的，其价格得分为满分，其他投标人的价格得分根据评标价格与评标基准价的偏离度计算，公式如下:
               - 当D1 = D * K时，F = 100;
               - 当D1 > D * K时，F = 100 - |D1 - D * K| ÷ (D * K) x 100 x E;
               - 当D1 < D * K时，F = 100 - |D1 - (D * K)| ÷ (D * K) x 100 x (E ÷ 2);
               其中，F为价格得分，D1为评标价格，D为评标基准价，E为减分系数（设置为1），K为区间系数（范围为90% - 99%，取值间隔为0.01）。
            2. 评标基准价计算:
               - 有效投标人数量 ≤ 5，D = 有效投标人评标价格的算术平均值；
               - 有效投标人数量 > 5，剔除1个最高评标价格和1个最低评标价格后，再剔除剩余有效投标人评标价格的算术平均值偏差20%（不含）以上的评标价格（如所有剩余有效投标人评标价格偏差均在20%（不含）以上，则不做剔除），D = 最终剩余有效投标人评标价格的算术平均值。
            3. 价格得分计算: 以各有效投标人投标一览表所填报的不含税总价进行评标，价格得分最低为0，最终计算结果按四舍五入保留小数点后两位
            """)
            K = st.number_input("区间系数 K (90% - 99%)", min_value=0.9, max_value=0.99, step=0.01, value=0.9)
        elif rule_choice == "4: 插入法计算得分":
            st.markdown("""
            一、评标基准价计算方法:
            评标价 Di:当通过初步评审的投标人数小于5个时，评标价为所有投标报价算术平均值。
            当通过初步评审的投标人数大于等于5个(含)时，评标价 Di为去掉1个最高和1个最低投标报价后的算术平均值。
            二、偏差率=100%*(投标人评标价 - 评标基准价)/评标基准价，偏差率保留 2位小数，示例0.00%。
            三、按插入法计算得分:
            Di > P:每高一个百分点从50分中扣1分;
            Di = P:得50分;
            Di < P: 每低一个百分点从50分中扣0.5分;
            扣完为止，得分四舍五入保留小数点后两位。
            其中:P表示评标基准价;Di表示投标函报价。
            """)
            K = None
        elif rule_choice == "5: 改进版插入法评分":
            st.markdown("""
            1. 所有符合评标基准价计算基数要求的有效投标报价去掉n个最高值和n个最低值后剩余投标人的有效投标报价算术平均值即为评标基准价P。(M为有效投标人的数量)
               当 M≤5时，n=0;
               当5<M≤10时，n=1;
               当 10<M≤20时，n=2;
               当M>20 时，n=3。
            2. 评分标准
               当投标人的投标报价等于评标基准价P时，得满分F分，投标报价每高于评标基准价P一个百分点扣a分;投标报价每低于评标基准价P一个百分点扣b分，扣完为止，得分按四舍五入，保留两位小数。投标报价得分计算方法如下:
               Px≦P: Sx=F-(P-Px)*b/P
               Px>P: Sx=F-(Px-P)*a/P
               其中:Sx为投标人投标报价的得分，F为投标报价部分满分，Px为投标人的评审价格，P为评标基准价。
            3. 所有计算过程数值及最终得分均四舍五入精确到小数点后两位，投标报价得分若为负分，则按零分处理。
            """)
            F = st.number_input("投标报价部分满分 F", min_value=0.0, step=0.01, value=100.0)
            a = st.number_input("投标报价每高于评标基准价P一个百分点的扣分数 a", min_value=0.0, step=0.01, value=1.0)
            b = st.number_input("投标报价每低于评标基准价P一个百分点的扣分数 b", min_value=0.0, step=0.01, value=0.5)

# 初始化 st.session_state
if 'my_discount' not in st.session_state:
    st.session_state.my_discount = 0.0
if 'my_price' not in st.session_state:
    st.session_state.my_price = 0.0

# 定义转换函数
def discount_to_price(discount, budget):
    if discount is not None:
        return discount / 100 * budget
    return None

def price_to_discount(price, budget):
    if price is not None:
        return price / budget * 100
    return None

# 主界面输入投标折扣
# 我方报价和折扣输入部分
if mode not in ["模式三：自动推荐我方折扣", "模式四：推荐我方和队友折扣"]:
    col1, col2 = st.columns(2)

    def update_my_price():
        global budget
        st.session_state.my_price = discount_to_price(st.session_state.my_discount, budget)

    def update_my_discount():
        global budget
        st.session_state.my_discount = price_to_discount(st.session_state.my_price, budget)

    with col1:
        # 初始化折扣值
        if 'my_discount' not in st.session_state:
            st.session_state.my_discount = 0.0
        my_discount = st.number_input(
            "我方折扣（%）",
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            key="my_discount",
            on_change=update_my_price
        )

    with col2:
        # 当最高限价更新时，不直接修改 st.session_state.my_price
        if 'prev_budget' not in st.session_state:
            st.session_state.prev_budget = budget
        if st.session_state.prev_budget != budget:
            st.session_state['rerun_flag'] = True
            st.session_state.prev_budget = budget

        # 初始化报价值，仅在第一次运行时设置
        if 'my_price' not in st.session_state:
            st.session_state.my_price = discount_to_price(st.session_state.my_discount, budget)

        my_price = st.number_input(
            "我方报价（万元）",
            min_value=0.0,
            max_value=budget,
            step=0.01,
            key="my_price",
            on_change=update_my_discount
        )

    if 'rerun_flag' in st.session_state and st.session_state['rerun_flag']:
        del st.session_state['rerun_flag']
        st.query_params = {"_rerun": "true"}

else:
    my_discount = None
    my_price = None

def discount_to_price(discount, budget):
    return discount / 100 * budget

def price_to_discount(price, budget):
    return price / budget * 100

# 初始化模拟器
simulator = BidSimulator(rule_dict[rule_choice], budget, price_weight)

# 先生成队友折扣填报窗口
if mode == "模式一：手动填写各方报价":
    simulator.generate_teammates(num_teammates)
    teammate_discounts = simulator.teammate_discounts
elif mode == "模式二：自动推荐队友折扣":
    teammate_discounts = None
elif mode == "模式三：自动推荐我方折扣":
    simulator.generate_teammates(num_teammates)
    teammate_discounts = simulator.teammate_discounts
elif mode == "模式四：推荐我方和队友折扣":
    teammate_discounts = None  # 不生成队友输入窗口

# 再生成对手折扣填报窗口
simulator.generate_competitors(num_competitors)

if st.button("开始模拟"):
    result_df = None  # 初始化result_df

    if mode == "模式四：推荐我方和队友折扣":
        competitor_ds = simulator.competitor_discounts
        my_discount, teammate_ds = simulator.recommend_my_and_teammates_discounts(
            competitor_ds, num_teammates, K, F, a, b
        )

        # 生成模拟结果
        my_price = my_discount / 100 * budget
        if rule_choice == "5: 改进版插入法评分":
            result_df = simulator.simulate(my_price, my_discount, teammate_ds, K, F, a, b)
        else:
            result_df = simulator.simulate(my_price, my_discount, teammate_ds, K)

        st.subheader("推荐结果")
        st.write(f"我方推荐折扣：{my_discount}%")
        for i, discount in enumerate(teammate_ds):
            st.write(f"队友 {i + 1} 推荐折扣：{discount}%")
    elif mode == "模式二：自动推荐队友折扣" and num_teammates > 0:
        recommended_discounts = simulator.recommend_teammate_discounts(
            my_price, my_discount,
            simulator.competitor_discounts,
            num_teammates,
            K,
            F,  # 新增参数
            a,  # 新增参数
            b  # 新增参数
        )
        st.subheader("自动推荐的队友折扣")
        for i, discount in enumerate(recommended_discounts):
            st.write(f"队友 {i + 1} 推荐折扣: {discount}%")
        teammate_discounts = recommended_discounts
        if rule_choice == "5: 改进版插入法评分":
            result_df = simulator.simulate(my_price, my_discount, teammate_discounts, K, F, a, b)
        else:
            result_df = simulator.simulate(my_price, my_discount, teammate_discounts, K)
    elif mode == "模式三：自动推荐我方折扣":
        # 获取已输入的队友和对手折扣
        teammate_ds = simulator.teammate_discounts
        competitor_ds = simulator.competitor_discounts

        # 计算推荐折扣
        rec_discount = simulator.recommend_my_discount(
            teammate_ds,
            competitor_ds,
            K,
            F,
            a,
            b
        )

        # 显示推荐结果
        st.subheader(f"自动推荐我方折扣: {rec_discount}%")

        # 用推荐折扣运行模拟
        my_price = rec_discount / 100 * budget
        if rule_choice == "5: 改进版插入法评分":
            result_df = simulator.simulate(my_price, rec_discount, teammate_ds, K, F, a, b)
        else:
            result_df = simulator.simulate(my_price, rec_discount, teammate_ds, K)
    else:
        if rule_choice == "5: 改进版插入法评分":
            result_df = simulator.simulate(my_price, my_discount, teammate_discounts, K, F, a, b)
        else:
            result_df = simulator.simulate(my_price, my_discount, teammate_discounts, K)

    # 确保在所有模式下都更新我方折扣信息
    if mode == "模式三：自动推荐我方折扣":
        my_index = result_df[result_df['单位类型'] == '我方'].index[0]
        result_df.at[my_index, '投标折扣（%）'] = rec_discount

    # 显示结果
    if result_df is not None:
        st.subheader("模拟结果")


        # 定义样式函数
        def apply_row_styles(row):
            style = []
            if row['_style'] == 'my_row':
                style = ['background-color: #B0E0E6; font-weight: bold'] * len(row)  # 浅蓝加粗
            elif row['_style'] == 'teammate_row':
                style = ['background-color: #B0E0E6'] * len(row)  # 浅蓝
            elif row['_style'] == 'opponent_row':
                style = ['background-color: #FFFFE0'] * len(row)  # 浅黄
            return style


        # 应用样式
        styled_df = (
            result_df.style
            .apply(apply_row_styles, axis=1)
            .highlight_max(
                subset=['得分'],
                props='color: #006400; font-weight: bold;'  # 深绿加粗显示最高分
            )
            .set_properties(**{
                'text-align': 'center',
                'white-space': 'pre'  # 保持列宽
            })
            .hide(axis='index')  # 隐藏索引
            .format({
                '投标折扣（%）': '{:.2f}%',
                '报价': '{:.2f}',
                '得分': '{:.2f}',
                '与我方得分分差': '{:+.2f}'
            })
        )

        # 显示优化后的表格
        st.dataframe(
            styled_df,
            width=800,
            height=min(400, 35 * (len(result_df) + 1))  # 动态调整高度
        )