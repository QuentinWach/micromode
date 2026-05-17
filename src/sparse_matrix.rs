use std::collections::BTreeMap;

use num_complex::Complex64;

#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl SparseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn eye(size: usize) -> Self {
        let values = vec![Complex64::new(1.0, 0.0); size];
        Self::diagonal(&values)
    }

    pub fn diagonal(values: &[Complex64]) -> Self {
        let mut triplets = Vec::with_capacity(values.len());
        for (index, value) in values.iter().copied().enumerate() {
            if value != Complex64::new(0.0, 0.0) {
                triplets.push((index, index, value));
            }
        }
        Self::from_triplets(values.len(), values.len(), triplets)
    }

    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: Vec<(usize, usize, Complex64)>,
    ) -> Self {
        let mut by_col = (0..cols)
            .map(|_| BTreeMap::<usize, Complex64>::new())
            .collect::<Vec<_>>();
        for (row, col, value) in triplets {
            assert!(row < rows);
            assert!(col < cols);
            if value == Complex64::new(0.0, 0.0) {
                continue;
            }
            *by_col[col].entry(row).or_insert(Complex64::new(0.0, 0.0)) += value;
        }
        Self::from_column_maps(rows, cols, by_col)
    }

    fn from_column_maps(rows: usize, cols: usize, by_col: Vec<BTreeMap<usize, Complex64>>) -> Self {
        assert_eq!(by_col.len(), cols);
        let mut col_ptrs = Vec::with_capacity(cols + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for col in by_col {
            for (row, value) in col {
                if value != Complex64::new(0.0, 0.0) {
                    row_indices.push(row);
                    values.push(value);
                }
            }
            col_ptrs.push(row_indices.len());
        }
        Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }

    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    pub fn shifted_diagonal(&self, shift: Complex64) -> Self {
        assert_eq!(self.rows, self.cols);
        let mut columns = (0..self.cols)
            .map(|_| BTreeMap::<usize, Complex64>::new())
            .collect::<Vec<_>>();
        for (col, column) in columns.iter_mut().enumerate().take(self.cols) {
            for (row, value) in self.column_entries(col) {
                column.insert(row, value);
            }
            *column.entry(col).or_insert(Complex64::new(0.0, 0.0)) -= shift;
        }
        Self::from_column_maps(self.rows, self.cols, columns)
    }

    pub fn column_entries(&self, col: usize) -> impl Iterator<Item = (usize, Complex64)> + '_ {
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        self.row_indices[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }

    pub fn scale(&self, scale: Complex64) -> Self {
        let mut out = self.clone();
        for value in &mut out.values {
            *value *= scale;
        }
        out
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let mut columns = (0..self.cols)
            .map(|_| BTreeMap::<usize, Complex64>::new())
            .collect::<Vec<_>>();
        for (col, column) in columns.iter_mut().enumerate().take(self.cols) {
            for (row, value) in self.column_entries(col) {
                *column.entry(row).or_insert(Complex64::new(0.0, 0.0)) += value;
            }
            for (row, value) in other.column_entries(col) {
                *column.entry(row).or_insert(Complex64::new(0.0, 0.0)) += value;
            }
        }
        Self::from_column_maps(self.rows, self.cols, columns)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(Complex64::new(-1.0, 0.0)))
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut columns = Vec::with_capacity(other.cols);
        for col in 0..other.cols {
            let mut accum = BTreeMap::<usize, Complex64>::new();
            for (inner, right) in other.column_entries(col) {
                for (row, left) in self.column_entries(inner) {
                    *accum.entry(row).or_insert(Complex64::new(0.0, 0.0)) += left * right;
                }
            }
            columns.push(accum);
        }
        Self::from_column_maps(self.rows, other.cols, columns)
    }

    pub fn matvec(&self, vector: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(self.cols, vector.len());
        let mut out = vec![Complex64::new(0.0, 0.0); self.rows];
        for (col, value) in vector.iter().copied().enumerate() {
            if value == Complex64::new(0.0, 0.0) {
                continue;
            }
            for (row, matrix_value) in self.column_entries(col) {
                out[row] += matrix_value * value;
            }
        }
        out
    }

    pub fn block_2x2(a: &Self, b: &Self, c: &Self, d: &Self) -> Self {
        assert_eq!(a.rows, b.rows);
        assert_eq!(c.rows, d.rows);
        assert_eq!(a.cols, c.cols);
        assert_eq!(b.cols, d.cols);
        let rows = a.rows + c.rows;
        let cols = a.cols + b.cols;
        let mut triplets = Vec::with_capacity(a.nnz() + b.nnz() + c.nnz() + d.nnz());
        append_block_triplets(&mut triplets, a, 0, 0);
        append_block_triplets(&mut triplets, b, 0, a.cols);
        append_block_triplets(&mut triplets, c, a.rows, 0);
        append_block_triplets(&mut triplets, d, a.rows, a.cols);
        Self::from_triplets(rows, cols, triplets)
    }

    pub fn block_grid(blocks: &[Vec<&Self>]) -> Self {
        assert!(!blocks.is_empty());
        assert!(!blocks[0].is_empty());
        let block_cols = blocks[0].len();
        assert!(blocks.iter().all(|row| row.len() == block_cols));

        let row_heights = blocks.iter().map(|row| row[0].rows).collect::<Vec<_>>();
        let col_widths = blocks[0].iter().map(|block| block.cols).collect::<Vec<_>>();
        for (row_index, row) in blocks.iter().enumerate() {
            for (col_index, block) in row.iter().enumerate() {
                assert_eq!(block.rows, row_heights[row_index]);
                assert_eq!(block.cols, col_widths[col_index]);
            }
        }

        let rows = row_heights.iter().sum();
        let cols = col_widths.iter().sum();
        let nnz = blocks
            .iter()
            .flat_map(|row| row.iter())
            .map(|block| block.nnz())
            .sum();
        let mut triplets = Vec::with_capacity(nnz);
        let mut row_offset = 0;
        for (row_index, row) in blocks.iter().enumerate() {
            let mut col_offset = 0;
            for (col_index, block) in row.iter().enumerate() {
                append_block_triplets(&mut triplets, block, row_offset, col_offset);
                col_offset += col_widths[col_index];
            }
            row_offset += row_heights[row_index];
        }
        Self::from_triplets(rows, cols, triplets)
    }
}

fn append_block_triplets(
    triplets: &mut Vec<(usize, usize, Complex64)>,
    matrix: &SparseMatrix,
    row_offset: usize,
    col_offset: usize,
) {
    for col in 0..matrix.cols {
        for (row, value) in matrix.column_entries(col) {
            triplets.push((row + row_offset, col + col_offset, value));
        }
    }
}
