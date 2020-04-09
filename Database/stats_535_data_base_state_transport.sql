-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: stats_535_data_base
-- ------------------------------------------------------
-- Server version	8.0.19

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `state_transport`
--

DROP TABLE IF EXISTS `state_transport`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `state_transport` (
  `STATE` text,
  `Airports` text,
  `Automobiles` text,
  `Buses` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `state_transport`
--

LOCK TABLES `state_transport` WRITE;
/*!40000 ALTER TABLE `state_transport` DISABLE KEYS */;
INSERT INTO `state_transport` VALUES ('Alabama','316','2,065,463','5,682'),('Alaska','767','173,487','8,080'),('American Samoa','','',''),('Arizona','298','2,391,624','9,051'),('Arkansas','304','926,789','11,941'),('California','882','14,860,967','99,917'),('Colorado','447','1,785,058','13,305'),('Connecticut','123','1,302,339','11,378'),('Delaware','42','419,199','3,763'),('District Of Columbia','16','208,141','4,885'),('Federated States Of Micronesia','','',''),('Florida','875','7,778,493','59,596'),('Georgia','454','3,536,055','36,418'),('Guam','4','',''),('Hawaii','46','512,386','3,033'),('Idaho','301','550,006','3,870'),('Illinois','707','4,655,844','34,359'),('Indiana','518','2,261,440','20,725'),('Iowa','279','1,268,996','9,284'),('Kansas','372','987,259','6,228'),('Kentucky','263','1,691,085','10,759'),('Louisiana','498','1,389,436','29,160'),('Maine','217','387,233','4,339'),('Marshall Islands','','',''),('Maryland','223','2,006,904','22,972'),('Massachusetts','225','2,208,468','13,498'),('Michigan','481','3,257,473','8,737'),('Minnesota','475','2,146,187','19,980'),('Mississippi','235','822,864','8,018'),('Missouri','510','2,141,050','27,180'),('Montana','301','441,392','4,702'),('Nebraska','246','677,040','13,564'),('Nevada','139','1,047,508','3,198'),('New Hampshire','186','505,381','2,871'),('New Jersey','320','2,798,699','25,097'),('New Mexico','171','609,494','6,278'),('New York','551','4,563,035','80,446'),('North Carolina','486','3,317,918','32,493'),('North Dakota','282','276,545','3,331'),('Northern Mariana Islands','','',''),('Ohio','660','4,605,459','41,255'),('Oklahoma','457','1,337,624','3,155'),('Oregon','442','1,542,461','17,691'),('Palau','','',''),('Pennsylvania','758','4,463,267','55,935'),('Puerto Rico','47','',''),('Rhode Island','22','420,313','2,273'),('South Carolina','196','1,808,401','17,728'),('South Dakota','185','353,180','2,614'),('Tennessee','339','2,325,004','29,158'),('Texas','2031','8,176,879','67,143'),('Utah','156','925,052','6,109'),('Vermont','89','221,936','1,070'),('Virgin Islands','','',''),('Virginia','431','3,251,339','34,268'),('Washington','538','2,925,765','23,566'),('West Virginia','125','562,984','3,342'),('Wisconsin','540','2,092,584','15,567'),('Wyoming','121','193,523','4,219');
/*!40000 ALTER TABLE `state_transport` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-04-09 16:55:12
