#!/bin/bash

# 创建安装目录
mkdir -p $HOME/java

# 下载JDK 17 (使用Amazon Corretto发行版，它是免费的)
wget -O $HOME/java/jdk17.tar.gz https://corretto.aws/downloads/latest/amazon-corretto-17-x64-linux-jdk.tar.gz

# 解压JDK
cd $HOME/java
tar -xzf jdk17.tar.gz
rm jdk17.tar.gz

# 获取解压后的目录名
JDK_DIR=$(ls -d */ | grep -i corretto)

# 添加环境变量配置到.bashrc
echo "\n# Java Environment Variables" >> $HOME/.bashrc
echo "export JAVA_HOME=\"$HOME/java/$JDK_DIR\"" >> $HOME/.bashrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> $HOME/.bashrc

# 使环境变量生效
source $HOME/.bashrc

# 验证安装
java -version