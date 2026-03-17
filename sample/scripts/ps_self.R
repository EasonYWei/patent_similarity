# =============================================================================
# ps_self.R - 专利级别自相似度计算
# =============================================================================
# 功能：计算每家公司的每个专利与其之前专利的余弦相似度
# 输出变量：
#   - sim_max: 最大余弦相似度
#   - sim_max_d: 与最大相似度专利的日期间隔（天）
#   - sim_ave: 平均余弦相似度
# =============================================================================

# 清除环境
rm(list = ls())

# 加载必要的包
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table")
}
library(data.table)

# -----------------------------------------------------------------------------
# 辅助函数：余弦相似度计算
# -----------------------------------------------------------------------------
# 计算两个向量之间的余弦相似度
cosine_similarity <- function(a, b) {
  sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))
}

# 计算一个向量与矩阵中所有向量的余弦相似度
cosine_similarity_matrix <- function(vec, mat) {
  # vec: 一个向量
  # mat: 矩阵，每行是一个向量
  numerator <- mat %*% vec
  norm_vec <- sqrt(sum(vec * vec))
  norm_mat <- sqrt(rowSums(mat * mat))
  as.vector(numerator / (norm_vec * norm_mat))
}

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

# 设置路径
input_file <- "output/patent_level_paraphrase_inspection.csv"
output_file <- "output/patent_self_similarity.csv"

# 读取数据
message("正在读取数据...")
dt <- fread(input_file, encoding = "UTF-8")
message(sprintf("共读取 %d 条专利记录", nrow(dt)))

# 识别 embedding 列（emb_0 到 emb_383）
emb_cols <- grep("^emb_\\d+$", names(dt), value = TRUE)
message(sprintf("识别到 %d 维 embedding", length(emb_cols)))

# 构造日期（这里使用年份的第一天作为日期，如果有具体日期请修改）
# 注意：数据中只有年份，我们假设为每年的1月1日
# 如果数据中有具体日期列，请替换下面的代码
dt[, date := as.Date(paste0(p_year, "-01-01"))]

# 按公司和日期排序
dt <- dt[order(stkcd, date, p_id)]

# 提取 embedding 矩阵
emb_matrix <- as.matrix(dt[, ..emb_cols])

# 初始化结果向量
n <- nrow(dt)
dt$sim_max <- NA_real_
dt$sim_max_d <- NA_real_
dt$sim_ave <- NA_real_

# 按公司分组计算
message("开始计算自相似度...")

unique_stkcd <- unique(dt$stkcd)
for (s in unique_stkcd) {
  # 获取该公司的索引
  idx <- which(dt$stkcd == s)
  n_company <- length(idx)
  
  message(sprintf("处理公司 %s，共有 %d 条专利记录", s, n_company))
  
  # 如果该公司只有一条专利，则无法计算相似度
  if (n_company < 2) {
    next
  }
  
  # 遍历该公司的每条专利（从第二条开始）
  for (i in 2:n_company) {
    current_idx <- idx[i]
    prev_indices <- idx[1:(i-1)]  # 之前的所有专利
    
    # 当前专利的 embedding
    current_emb <- emb_matrix[current_idx, ]
    
    # 之前所有专利的 embedding 矩阵
    prev_emb <- emb_matrix[prev_indices, , drop = FALSE]
    
    # 计算余弦相似度
    similarities <- cosine_similarity_matrix(current_emb, prev_emb)
    
    # 计算统计量
    max_sim <- max(similarities)
    max_idx <- prev_indices[which.max(similarities)]
    avg_sim <- mean(similarities)
    
    # 计算日期间隔（天）
    days_diff <- as.numeric(dt$date[current_idx] - dt$date[max_idx])
    
    # 存储结果
    dt$sim_max[current_idx] <- max_sim
    dt$sim_max_d[current_idx] <- days_diff
    dt$sim_ave[current_idx] <- avg_sim
  }
}

# 选择输出列
result <- dt[, .(stkcd, p_year, p_id, date, sim_max, sim_max_d, sim_ave)]

# 保存结果
fwrite(result, output_file, row.names = FALSE)
message(sprintf("结果已保存至: %s", output_file))

# 输出统计摘要
message("\n=== 统计摘要 ===")
message(sprintf("总专利数: %d", nrow(result)))
message(sprintf("有相似度计算的专利数: %d", sum(!is.na(result$sim_max))))
message(sprintf("sim_max - 均值: %.4f, 中位数: %.4f, 范围: [%.4f, %.4f]",
                mean(result$sim_max, na.rm = TRUE),
                median(result$sim_max, na.rm = TRUE),
                min(result$sim_max, na.rm = TRUE),
                max(result$sim_max, na.rm = TRUE)))
message(sprintf("sim_ave - 均值: %.4f, 中位数: %.4f, 范围: [%.4f, %.4f]",
                mean(result$sim_ave, na.rm = TRUE),
                median(result$sim_ave, na.rm = TRUE),
                min(result$sim_ave, na.rm = TRUE),
                max(result$sim_ave, na.rm = TRUE)))
message(sprintf("sim_max_d - 均值: %.1f 天, 中位数: %.1f 天, 范围: [%d, %d] 天",
                mean(result$sim_max_d, na.rm = TRUE),
                median(result$sim_max_d, na.rm = TRUE),
                min(result$sim_max_d, na.rm = TRUE),
                max(result$sim_max_d, na.rm = TRUE)))

# 显示前几行结果
message("\n=== 前10行结果预览 ===")
print(head(result, 10))
