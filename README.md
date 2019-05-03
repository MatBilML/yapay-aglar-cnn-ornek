# Giriş

Bu proje CNN - Inception v3 kullanılarak kedi ve köpekleri sınıflandırır. Kullanılan verisetine [buraya](https://www.kaggle.com/chetankv/dogs-cats-images) tıklayarak ulaşabilirsiniz.


# Gereklilikler
Python 3.0+
Tensorflow 1.0+

Gerekli kütüphaneleri bu kod bloğu ile yükleyebilirsiniz:

    pip install -r requirements.txt

# Eğitim
Eğitim için gerekli kod bloğu aşağıdaki gibidir. Komut istemi üzerinden proje klasörü içerisindeyken dilediğiniz değişiklikleri yaparak çalıştırabilirsiniz. Eğitimde kullanılacak verilerin klasörünü --image_dir argümanına, eğitim adım sayısını da --how_many_training_steps argümanına verdikten sonra eğitimi başlatabilirsiniz.

    python train.py --bottleneck_dir=log/bottlenecks --how_many_training_steps=7000 --model_dir=inception --summaries_dir=log/training_summaries/basic --output_graph=log/trained_graph.pb --output_labels=log/trained_labels.txt --image_dir=./train



# Test

Test klasöründeki verileri test etmek için test.py dosyasını çalıştırabilirsiniz.