class VeriModelTakviye:
    @staticmethod
    def dataframe_to_scores(
        df: Any,
        score_columns: List[str],
        dataframe_type: str = 'auto'
    ) -> Dict[str, float]:
        """
        Farklı dataframe türlerinden (pandas, polars, numpy) skor sözlüğüne dönüştürür.

        Args:
            df: pandas.DataFrame, polars.DataFrame veya numpy.ndarray
            score_columns: Kullanılacak skor kolon isimleri
            dataframe_type: 'pandas', 'polars', 'numpy', 'auto' (otomatik algılama)
        Returns:
            Dict[str, float]: Her skor kolonu için son satır değerlerinden oluşan sözlük
        """
        scores = {}

        # ✅ 1. DataFrame türünü tespit et
        if dataframe_type == 'auto':
            if hasattr(df, 'iloc') and hasattr(df, 'columns'):  # pandas
                dataframe_type = 'pandas'
            elif hasattr(df, 'select') and hasattr(df, 'schema'):  # polars
                dataframe_type = 'polars'
            elif isinstance(df, np.ndarray):
                dataframe_type = 'numpy'
            else:
                raise ValueError(f"Unsupported dataframe type: {type(df)}")

        try:
            # ✅ 2. Pandas tipi işleme
            if dataframe_type == 'pandas':
                for col in score_columns:
                    if col in df.columns:
                        scores[col] = float(df[col].iloc[-1])  # Son değeri al
                    else:
                        logger.warning(f"Pandas kolon bulunamadı: {col}")
                        scores[col] = 0.5  # Fallback

            # ✅ 3. Polars tipi işleme
            elif dataframe_type == 'polars':
                for col in score_columns:
                    if col in df.columns:
                        scores[col] = float(df[col][-1])  # Son değeri al
                    else:
                        logger.warning(f"Polars kolon bulunamadı: {col}")
                        scores[col] = 0.5

            # ✅ 4. Numpy tipi işleme
            elif dataframe_type == 'numpy':
                if len(score_columns) != df.shape[1]:
                    raise ValueError(
                        f"Column count mismatch: {len(score_columns)} names, {df.shape[1]} columns"
                    )
                for i, col in enumerate(score_columns):
                    scores[col] = float(df[-1, i])  # Son satırdan değer al

            return scores

        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Dataframe conversion error: {e}")
            # ✅ Tüm kolonlara default skor ver (örneğin 0.5)
            return {col: 0.5 for col in score_columns}
