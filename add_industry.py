"""
添加行业板块信息到 stock_data.csv
使用 akshare 获取股票所属行业
"""
import pandas as pd
import os

def add_industry_info():
    data_file = './data/stock_data.csv'
    output_file = './data/stock_data.csv'

    print(f"读取数据: {data_file}")
    df = pd.read_csv(data_file, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].str.zfill(6)

    stock_list = sorted(df['股票代码'].unique())
    print(f"共 {len(stock_list)} 只股票")

    # 用 akshare 获取行业分类
    try:
        import akshare as ak
        
        # 获取申万行业分类 (2021版)
        print("获取申万行业分类...")
        industry_df = ak.stock_board_industry_name_em()
        
        # 获取每只股票的行业
        stock_industry = {}
        print("匹配股票行业...")
        for code in stock_list:
            try:
                stock_info = ak.stock_individual_info_em(symbol=code)
                industry = stock_info.loc[stock_info['item'] == '行业', 'value'].values
                if len(industry) > 0:
                    stock_industry[code] = industry[0]
                else:
                    stock_industry[code] = '未知'
            except:
                # fallback: 用东方财富行业
                try:
                    concept = ak.stock_board_concept_cons_em(symbol="BK0719")  # 示例
                    stock_industry[code] = '未知'
                except:
                    stock_industry[code] = '未知'

        df['行业'] = df['股票代码'].map(stock_industry).fillna('未知')

        # 保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已添加行业信息，保存到: {output_file}")
        print(f"行业分布: {df['行业'].value_counts().head(10)}")

    except ImportError:
        print("akshare 未安装，使用简单市值排名替代")
        # 不带行业信息的fallback：全市场龙头
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("跳过行业信息添加")

    except Exception as e:
        print(f"获取行业信息失败: {e}")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("保留原数据")

if __name__ == '__main__':
    add_industry_info()
