import pandas as pd
from dbfread import DBF


def main():
    try:
        dbf = DBF('db_var_files/pokorny.dbf', load=True, encoding='mbcs')
    except Exception as err:
        print(err)
        breakpoint()
    df = pd.DataFrame(iter(dbf))
    print(df.iloc[0])
    breakpoint()
    pass


if __name__ == '__main__':
    main()
    pass
