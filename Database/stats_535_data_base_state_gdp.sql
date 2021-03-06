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
-- Table structure for table `state_gdp`
--

DROP TABLE IF EXISTS `state_gdp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `state_gdp` (
  `State` text,
  `gdpRank` int DEFAULT NULL,
  `stateGDP` int DEFAULT NULL,
  `stateGDPperc` double DEFAULT NULL,
  `gdpGrowth2018` double DEFAULT NULL,
  `Pop` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `state_gdp`
--

LOCK TABLES `state_gdp` WRITE;
/*!40000 ALTER TABLE `state_gdp` DISABLE KEYS */;
INSERT INTO `state_gdp` VALUES ('California',1,3018337,0.145,3.5,39937489),('Texas',2,1818585,0.087,3.2,29472295),('New York',3,1701399,0.082,2.1,19440469),('Florida',4,1059144,0.051,3.5,21992985),('Illinois',5,879947,0.042,2.1,12659682),('Pennsylvania',6,803307,0.038,2.1,12820878),('Ohio',7,689139,0.033,1.8,11747694),('New Jersey',8,634721,0.03,2,8936574),('Georgia',9,601503,0.029,2.6,10736059),('Washington',10,576624,0.028,5.7,7797095),('Massachusetts',11,575635,0.028,2.7,6976597),('North Carolina',12,575605,0.028,2.9,10611862),('Virginia',13,544348,0.026,2.8,8626207),('Michigan',14,537087,0.026,2.7,10045029),('Maryland',15,417776,0.02,1.6,6083116),('Colorado',16,376994,0.018,3.5,5845526),('Minnesota',17,374920,0.018,2.2,5700671),('Tennessee',18,373663,0.018,3,6897576),('Indiana',19,371629,0.018,1.9,6745354),('Arizona',20,355311,0.017,4,7378494),('Wisconsin',21,342470,0.016,2.5,5851754),('Missouri',22,323287,0.015,2.3,6169270),('Connecticut',23,279653,0.013,1,3563077),('Louisiana',24,255492,0.012,1.1,4645184),('Oregon',25,243085,0.012,3.4,4301089),('South Carolina',26,234367,0.011,1.6,5210095),('Alabama',27,224654,0.011,2,4908621),('Kentucky',28,211621,0.01,1.4,4499692),('Oklahoma',29,203250,0.01,1.8,3954821),('Iowa',30,192608,0.009,1.4,3179849),('Utah',31,180862,0.009,4.3,3282115),('Kansas',32,169558,0.008,1.9,2910357),('Nevada',33,168752,0.008,3.2,3139658),('Arkansas',34,129812,0.006,0.9,3038999),('Nebraska',35,124742,0.006,1.5,1952570),('Mississippi',36,115749,0.006,1,2989260),('New Mexico',37,101452,0.005,1.8,2096640),('Hawaii',38,93419,0.004,1,1412687),('New Hampshire',39,86046,0.004,2.2,1371246),('West Virginia',40,79168,0.004,2.4,1778070),('Idaho',41,78640,0.004,4.1,1826156),('Delaware',42,76537,0.004,0.3,982895),('Maine',43,65349,0.003,1.9,1345790),('Rhode Island',44,61341,0.003,0.6,1056161),('North Dakota',45,55657,0.003,2.5,761723),('Alaska',46,54851,0.003,-0.3,734002),('South Dakota',47,52544,0.003,1.3,903027),('Montana',48,49635,0.002,0.9,1086759),('Wyoming',49,39899,0.002,0.3,567025),('Vermont',50,34154,0.002,1.2,628061);
/*!40000 ALTER TABLE `state_gdp` ENABLE KEYS */;
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
