1. .gitignore
2. .dockerignore
3. Dockerfile
4. .github/workflows/docker-deploy.yml


===
✅ 1. .gitignore
Amaç: Git deposuna dahil edilmeyecek dosyaları tanımlar.
Neden ilk? Projeye başlarken geçici dosyalar, derleme çıktıları vb. şeylerin yanlışlıkla Git'e eklenmemesi için ilk başta oluşturulmalı.

✅ 2. .dockerignore
Amaç: Docker image'ı oluşturulurken konteynere dahil edilmeyecek dosyaları tanımlar.
Neden ikinci? .gitignore gibi, ama Docker bağlamında çalışır. Gereksiz dosyaların image'a girmesini önlemek için erken oluşturulmalı.

✅ 3. Dockerfile
Amaç: Docker image'ını nasıl oluşturacağını tanımlar.
Neden üçüncü? Bu dosya olmadan CI/CD ya da manuel olarak Docker image'ı oluşturulamaz. .dockerignore hazır olduğunda bu dosyada tanımlanan bağlam doğru çalışır.

✅ 4. .github/workflows/docker-deploy.yml
Amaç: GitHub Actions içinde Docker image'ını oluşturma, test etme ve dağıtma gibi işlemleri otomatik hale getirir.
Neden son? CI/CD süreci için Dockerfile ve diğer tüm yapılandırma dosyalarının hazır olması gerekir. Workflow en son yazılır ki eksik dosya hataları alınmasın.
