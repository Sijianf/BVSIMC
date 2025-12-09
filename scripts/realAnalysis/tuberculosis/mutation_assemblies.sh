mkdir -p assemblies
cd assemblies

cut -d',' -f2 ../all_assemblies.csv | tail -n +2 > all_accs.txt


while read acc; do
  echo "\n========== Downloading $acc ==========\n"
  datasets download genome accession "$acc" --filename "${acc}.zip" 
  unzip -q -o "${acc}.zip" -d "${acc}_dir"
  fasta=$(find "${acc}_dir" -type f -iname "*genomic.fna" -o -iname "*genomic.fasta" | head -n1)

  if [ -n "$fasta" ]; then
    cp "$fasta" "${acc}.fna"
    echo "[OK] Saved ${acc}.fna"
  else
    echo "[WARNING] FASTA not found for $acc"
    sleep 5
  fi

done < all_accs.txt


# 下载参考基因组 FASTA
curl -o H37Rv.fna.gz \
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz

# 下载参考基因组 GFF 注释
curl -o H37Rv.gff.gz \
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.gff.gz

# 解压
gunzip H37Rv.fna.gz
gunzip H37Rv.gff.gz


brew install minimap2
brew install samtools
brew install bcftools
brew install parallel

/opt/homebrew/Cellar/parallel/20251122
/opt/homebrew/share/zsh/site-functions
pip install cyvcf2


# minimap2 index
minimap2 -d H37Rv.mmi H37Rv.fna

# samtools index
samtools faidx H37Rv.fna

cd assemblies

ACC=GCA_014489215.1

# 1）mapping + sorting
minimap2 -a ../reference/H37Rv.mmi ${ACC}.fna | \
    samtools sort -o ${ACC}.sorted.bam

# 2）index BAM
samtools index ${ACC}.sorted.bam

# 3）call SNP (generate VCF)
bcftools mpileup -f ../reference/H37Rv.fna ${ACC}.sorted.bam -a FORMAT/DP -Ou | \
    bcftools call -mv -Oz -o ${ACC}.raw.vcf.gz

# 4）index VCF
bcftools index ${ACC}.raw.vcf.gz

# 5）preview VCF
bcftools view ${ACC}.raw.vcf.gz | sed -n '1,20p'



git clone https://github.com/GTB-tbsequencing/mutation-catalogue-2023.git
cd mutation-catalogue-2023/Final\ Result\ Files
ls -lh



# 使用parallel生成VCF
mkdir -p logs

cd ..

ls assemblies/*.fna | wc -l

parallel --halt now,fail=1 -a assemblies/all_accs.txt -j 10 --bar
# parallel -a assemblies/all_accs.txt -j 10 --bar '
ACC={};
echo "\n=== START ${ACC} $(date) ===\n" >> logs/batch.log
# 如果已经生成 VCF 则跳过
if [ -f "assemblies/${ACC}.raw.vcf.gz" ]; then
  echo "[SKIP] ${ACC} already has VCF" >> logs/batch.log
  exit 0
fi
set -o pipefail

# map -> sort -> index -> call VCF (文件读/写都在 assemblies/)
minimap2 -a reference/H37Rv.mmi assemblies/${ACC}.fna \
  | samtools sort -o assemblies/${ACC}.sorted.bam

samtools index assemblies/${ACC}.sorted.bam

bcftools mpileup -f reference/H37Rv.fna assemblies/${ACC}.sorted.bam -a FORMAT/DP -Ou \
  | bcftools call -mv -Oz -o assemblies/${ACC}.raw.vcf.gz

bcftools index assemblies/${ACC}.raw.vcf.gz

if [ $? -eq 0 ]; then
  echo "[OK] ${ACC} $(date)" >> logs/batch.log
  # 删除中间 BAM 释放空间
  rm -f assemblies/${ACC}.sorted.bam assemblies/${ACC}.sorted.bam.bai
else
  echo "[FAIL] ${ACC} $(date)" >> logs/batch.log
fi
'

