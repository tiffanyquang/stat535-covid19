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
-- Table structure for table `states_by_color`
--

DROP TABLE IF EXISTS `states_by_color`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `states_by_color` (
  `State` text,
  `pvi` int DEFAULT NULL,
  `governorParty` text,
  `senateParty` text,
  `houseBalance` text,
  `Pop` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `states_by_color`
--

LOCK TABLES `states_by_color` WRITE;
/*!40000 ALTER TABLE `states_by_color` DISABLE KEYS */;
INSERT INTO `states_by_color` VALUES ('Hawaii',-18,'Democratic','Democratic','2D',1412687),('Vermont',-15,'Republican','Democratic*','1D',628061),('California',-12,'Democratic','Democratic','46D, 7R',39937489),('Maryland',-12,'Republican','Democratic','7D, 1R',6083116),('Massachusetts',-12,'Republican','Democratic','9D',6976597),('New York',-12,'Democratic','Democratic','21D, 6R',19440469),('Rhode Island',-10,'Democratic','Democratic','2D',1056161),('Illinois',-7,'Democratic','Democratic','13D, 5R',12659682),('New Jersey',-7,'Democratic','Democratic','11D, 1R',8936574),('Washington',-7,'Democratic','Democratic','7D, 3R',7797095),('Connecticut',-6,'Democratic','Democratic','5D',3563077),('Delaware',-6,'Democratic','Democratic','1D',982895),('Oregon',-5,'Democratic','Democratic','4D, 1R',4301089),('Maine',-3,'Democratic','Both*','2D',1345790),('New Mexico',-3,'Democratic','Democratic','3D',2096640),('Colorado',-1,'Democratic','Both','4D, 3R',5845526),('Michigan',-1,'Democratic','Democratic','7D, 6R, 1I',10045029),('Minnesota',-1,'Democratic','Democratic','5D, 3R',5700671),('Nevada',-1,'Democratic','Democratic','3D, 1R',3139658),('Virginia',-1,'Democratic','Democratic','7D, 4R',8626207),('New Hampshire',0,'Republican','Democratic','2D',1371246),('Pennsylvania',0,'Democratic','Both','9D, 9R',12820878),('Wisconsin',0,'Democratic','Both','5R, 3D',5851754),('Florida',2,'Republican','Republican','14R, 13D',21992985),('Iowa',3,'Republican','Republican','3D, 1R',3179849),('North Carolina',3,'Democratic','Republican','8R, 3D, 2 Vacant',10611862),('Ohio',3,'Republican','Both','12R, 4D',11747694),('Arizona',5,'Republican','Both','5D, 4R',7378494),('Georgia',5,'Republican','Republican','9R, 5D',10736059),('South Carolina',8,'Republican','Republican','5R, 2D',5210095),('Texas',8,'Republican','Republican','23R, 13D',29472295),('Alaska',9,'Republican','Republican','1R',734002),('Indiana',9,'Republican','Republican','7R, 2D',6745354),('Mississippi',9,'Republican','Republican','3R, 1D',2989260),('Missouri',9,'Republican','Republican','6R, 2D',6169270),('Louisiana',11,'Democratic','Republican','5R, 1D',4645184),('Montana',11,'Democratic','Both','1R',1086759),('Kansas',13,'Democratic','Republican','3R, 1D',2910357),('Alabama',14,'Republican','Both','6R, 1D',4908621),('Nebraska',14,'Republican','Republican','3R',1952570),('South Dakota',14,'Republican','Republican','1R',903027),('Tennessee',14,'Republican','Republican','7R, 2D',6897576),('Arkansas',15,'Republican','Republican','4R',3038999),('Kentucky',15,'Republican','Republican','5R, 1D',4499692),('North Dakota',17,'Republican','Republican','1R',761723),('Idaho',19,'Republican','Republican','2R',1826156),('West Virginia',19,'Republican','Both','3R',1778070),('Oklahoma',20,'Republican','Republican','4R, 1D',3954821),('Utah',20,'Republican','Republican','3R, 1D',3282115),('Wyoming',25,'Republican','Republican','1R',567025);
/*!40000 ALTER TABLE `states_by_color` ENABLE KEYS */;
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
